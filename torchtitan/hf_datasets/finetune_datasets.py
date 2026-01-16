# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Finetuning datasets for instruction/chat data.

This module provides:
1. PackedSequencesDataset - Packs multiple conversations into sequences for efficiency
2. SingleSequenceDataset - One conversation per sequence (for long-context training)
3. Data streaming utilities for HuggingFace and local JSONL datasets
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from typing import Any, TypeVar

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

from .chat_template import (
    ChatTemplate,
    HfTokenizerWithPadToken,
    MessageRow,
    TokensAndMask,
    tokenize_and_mask,
    tokenize_no_mask,
)

# Enable debug tracing with environment variable: TORCHTITAN_DEBUG_TOKENS=1
DEBUG_TOKENS = os.environ.get("TORCHTITAN_DEBUG_TOKENS", "0") == "1"
# Counter to limit debug output
_DEBUG_BATCH_COUNT = 0
_DEBUG_MAX_BATCHES = int(os.environ.get("TORCHTITAN_DEBUG_MAX_BATCHES", "5"))

T = TypeVar("T")
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT")


class StreamableData(Iterable[T]):
    """Base class for streamable data sources with seek support."""

    def __iter__(self) -> Iterator[T]:
        ...

    def seek(self, idx: int):
        ...

    def map(self, func: Callable[[T], OutputT]) -> "StreamableData[OutputT]":
        return MappedStream(func, self)


class HuggingFaceStream(StreamableData[dict[str, Any]]):
    """Stream data from a HuggingFace dataset with distributed support."""

    def __init__(self, hf_dataset: Any, infinite: bool = False):
        self.hf_dataset = hf_dataset
        self.infinite = infinite
        self.start_idx = 0

    @classmethod
    def from_local_jsonl(
        cls,
        dataset_path: str,
        split: str = "train",
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> "HuggingFaceStream":
        """Load from a local JSONL file."""
        ds = load_dataset("json", data_files=dataset_path, split=split, streaming=True)
        data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        return HuggingFaceStream(data, infinite=infinite)

    @classmethod
    def from_hf(
        cls,
        dataset_path: str,
        split: str = "train",
        dp_rank: int = 0,
        dp_world_size: int = 1,
        token: str | None = None,
        name: str | None = None,
        data_files: str | None = None,
        infinite: bool = False,
    ) -> "HuggingFaceStream":
        """Load from a HuggingFace dataset."""
        ds = load_dataset(
            dataset_path,
            name=name,
            data_files=data_files,
            split=split,
            streaming=True,
            token=token,
        )
        data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        return HuggingFaceStream(data, infinite=infinite)

    def _get_data_iter(self):
        """Get iterator, handling resume for map-style datasets."""
        if isinstance(self.hf_dataset, Dataset):
            if self.start_idx >= len(self.hf_dataset):
                return iter([])
            else:
                return iter(self.hf_dataset.skip(self.start_idx))
        return iter(self.hf_dataset)

    def seek(self, idx: int):
        """Seek to a specific position for checkpointing."""
        self.start_idx = idx

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.infinite:
            while True:
                yield from self._get_data_iter()
        else:
            yield from self._get_data_iter()


class MappedStream(StreamableData[T]):
    """A stream that applies a transformation function to each element."""

    def __init__(
        self, func: Callable[[InputT], T], input_stream: StreamableData[InputT]
    ):
        self.func = func
        self.input_stream = input_stream

    def __iter__(self) -> Iterator[T]:
        for elem in self.input_stream:
            yield self.func(elem)

    def seek(self, idx: int):
        return self.input_stream.seek(idx)


def extract_text(item: dict[str, Any]) -> str:
    """Extract text from a dataset item."""
    return item["text"]


def extract_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract messages from a dataset item."""
    return item["messages"]


class SingleSequenceDataset(IterableDataset, Stateful):
    """Dataset that yields individual sequences without packing.

    Each document is padded to seq_len if shorter, or dropped if longer.
    Used for long-context training with context parallelism where document
    packing is not desirable.

    Args:
        data_stream: Stream of TokensAndMask objects
        seq_len: Maximum sequence length
        pad_token_id: Token ID to use for padding
    """

    def __init__(
        self,
        data_stream: StreamableData[TokensAndMask],
        seq_len: int,
        pad_token_id: int,
    ) -> None:
        self._data_stream = data_stream
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = -100  # Standard ignore index for loss
        self._next_idx = 0
        self._logged_too_long = False
        self._logged_no_trainable = False

    def __iter__(self):
        global _DEBUG_BATCH_COUNT
        max_token_len = 1 + self.seq_len  # +1 for labels shift

        for transcript_and_mask in self._data_stream:
            incoming_len = len(transcript_and_mask.tokens)

            if incoming_len > max_token_len:
                if not self._logged_too_long:
                    logger.warning(
                        f"Dataset contains sequences exceeding max length {max_token_len}. "
                        f"These will be dropped. First seen: {incoming_len} tokens."
                    )
                    self._logged_too_long = True
                continue

            if incoming_len == 0:
                continue

            # Pad to max_token_len
            padding_len = max_token_len - incoming_len
            tokens = torch.LongTensor(
                list(
                    itertools.chain(
                        transcript_and_mask.tokens,
                        [self.pad_token_id] * padding_len,
                    )
                )
            )
            masks = torch.BoolTensor(
                list(
                    itertools.chain(
                        transcript_and_mask.mask,
                        [False] * padding_len,
                    )
                )
            )

            # Create labels: shift by 1 and mask non-trainable tokens
            labels = tokens[1:].clone().detach()
            labels[~masks[1:]] = self.mask_token_id

            # Validate that we have trainable tokens
            trainable_count = masks[1:].sum().item()
            if trainable_count == 0:
                if not self._logged_no_trainable:
                    logger.warning(
                        "Dataset contains sequences with no trainable tokens. "
                        "These will be skipped to avoid NaN loss."
                    )
                    self._logged_no_trainable = True
                continue

            # Debug tracing
            if DEBUG_TOKENS and _DEBUG_BATCH_COUNT < _DEBUG_MAX_BATCHES:
                _DEBUG_BATCH_COUNT += 1
                self._log_debug_info(
                    "SingleSequenceDataset", tokens, labels, masks, trainable_count
                )

            self._next_idx += 1
            yield {"input": tokens[:-1]}, labels

    def _log_debug_info(
        self,
        dataset_type: str,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        trainable_count: int,
    ):
        """Log debug information about the batch."""
        global _DEBUG_BATCH_COUNT
        logger.warning("=" * 80)
        logger.warning(
            "[DEBUG DATALOADER] %s yielding batch %d", dataset_type, _DEBUG_BATCH_COUNT
        )
        logger.warning("  Input tokens shape: %s", tokens[:-1].shape)
        logger.warning("  Labels shape: %s", labels.shape)
        logger.warning("  Trainable tokens: %d / %d", trainable_count, len(labels))

        # Count IGNORE_INDEX (-100) in labels
        ignored_count = (labels == self.mask_token_id).sum().item()
        valid_count = (labels != self.mask_token_id).sum().item()
        logger.warning(
            "  Labels: %d valid, %d ignored (IGNORE_INDEX=-100)",
            valid_count,
            ignored_count,
        )

        # Check for special tokens in labels
        pad_in_labels = (labels == self.pad_token_id).sum().item()
        logger.warning(
            "  PAD token (%d) appears in labels: %d times",
            self.pad_token_id,
            pad_in_labels,
        )

        # Show last 15 labels
        logger.warning("  Last 15 positions (input -> label, trainable):")
        for i in range(max(0, len(labels) - 15), len(labels)):
            input_tok = tokens[i].item()
            label_tok = labels[i].item()
            is_trainable = labels[i].item() != self.mask_token_id
            logger.warning(
                "    [%d] input=%d -> label=%d, trainable=%s",
                i,
                input_tok,
                label_tok,
                is_trainable,
            )
        logger.warning("=" * 80)

    def load_state_dict(self, state_dict):
        self._next_idx = state_dict["_next_idx"]
        self._data_stream.seek(self._next_idx + 1)

    def state_dict(self):
        return {"_next_idx": self._next_idx}


class PackedSequencesDataset(IterableDataset, Stateful):
    """Dataset that packs multiple sequences into fixed-length chunks.

    This is more efficient than SingleSequenceDataset as it minimizes padding
    by concatenating multiple conversations together.

    Args:
        data_stream: Stream of TokensAndMask objects
        seq_len: Maximum sequence length
        split_transcripts: Whether to split long documents across chunks
        pad_token_id: Token ID to use for padding
    """

    def __init__(
        self,
        data_stream: StreamableData[TokensAndMask],
        seq_len: int,
        split_transcripts: bool,
        pad_token_id: int,
    ) -> None:
        self._data_stream = data_stream
        self.seq_len = seq_len
        self.split_tokens = split_transcripts
        self.pad_token_id = pad_token_id
        self.mask_token_id = -100
        self._next_idx = 0
        self._logged_too_long = False
        self._logged_no_trainable = False

    def __iter__(self):
        global _DEBUG_BATCH_COUNT
        max_buffer_token_len = 1 + self.seq_len
        outgoing_transcripts: list[TokensAndMask] = []

        def packed_len():
            return sum(len(tandm.tokens) for tandm in outgoing_transcripts)

        for transcript_and_mask in self._data_stream:
            incoming_len = len(transcript_and_mask.tokens)

            # Drop if too long and we're not splitting
            if incoming_len > max_buffer_token_len and not self.split_tokens:
                if not self._logged_too_long:
                    logger.warning(
                        f"Dataset contains sequences exceeding max length {max_buffer_token_len}. "
                        f"These will be dropped. First seen: {incoming_len} tokens."
                    )
                    self._logged_too_long = True
                continue

            # Drain buffer when it would overflow
            while len(transcript_and_mask.tokens) + packed_len() > max_buffer_token_len:
                if self.split_tokens:
                    prefix, transcript_and_mask = transcript_and_mask.split(
                        max_buffer_token_len - packed_len()
                    )
                    outgoing_transcripts.append(prefix)

                # Create the output batch
                padding_len = max_buffer_token_len - packed_len()
                tokens = torch.LongTensor(
                    list(
                        itertools.chain(
                            *(t.tokens for t in outgoing_transcripts),
                            [self.pad_token_id] * padding_len,
                        )
                    )
                )
                masks = torch.BoolTensor(
                    list(
                        itertools.chain(
                            *(t.mask for t in outgoing_transcripts),
                            [False] * padding_len,
                        )
                    )
                )

                if logger.isEnabledFor(10):  # DEBUG level
                    logger.debug(
                        "Sending %s documents out. %s bytes in transcript, "
                        "%s pad tokens, %s trainable tokens, mask shape = %s",
                        len(outgoing_transcripts),
                        packed_len(),
                        padding_len,
                        masks.count_nonzero(),
                        masks.shape,
                    )

                outgoing_transcripts.clear()

                # Create labels
                labels = tokens[1:].clone().detach()
                labels[~masks[1:]] = self.mask_token_id

                # Validate trainable tokens
                trainable_count = masks[1:].sum().item()
                if trainable_count == 0:
                    if not self._logged_no_trainable:
                        logger.warning(
                            "Dataset contains batches with no trainable tokens. "
                            "These will be skipped to avoid NaN loss."
                        )
                        self._logged_no_trainable = True
                    continue

                # Debug tracing
                if DEBUG_TOKENS and _DEBUG_BATCH_COUNT < _DEBUG_MAX_BATCHES:
                    _DEBUG_BATCH_COUNT += 1
                    self._log_debug_info(tokens, labels, masks, trainable_count)

                self._next_idx += 1
                yield {"input": tokens[:-1]}, labels

            # Add remaining tokens to buffer
            if len(transcript_and_mask.tokens):
                outgoing_transcripts.append(transcript_and_mask)

    def _log_debug_info(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        trainable_count: int,
    ):
        """Log debug information about the batch."""
        global _DEBUG_BATCH_COUNT
        logger.warning("=" * 80)
        logger.warning(
            "[DEBUG DATALOADER] PackedSequencesDataset yielding batch %d",
            _DEBUG_BATCH_COUNT,
        )
        logger.warning("  Input tokens shape: %s", tokens[:-1].shape)
        logger.warning("  Labels shape: %s", labels.shape)
        logger.warning(
            "  Trainable tokens in mask: %d / %d", trainable_count, len(labels)
        )

        # Count IGNORE_INDEX in labels
        ignored_count = (labels == self.mask_token_id).sum().item()
        valid_count = (labels != self.mask_token_id).sum().item()
        logger.warning(
            "  Labels: %d valid, %d ignored (IGNORE_INDEX=-100)",
            valid_count,
            ignored_count,
        )

        # Check for special tokens in labels
        pad_in_labels = (labels == self.pad_token_id).sum().item()
        logger.warning(
            "  PAD token (%d) appears in labels: %d times",
            self.pad_token_id,
            pad_in_labels,
        )

        # Show last 20 labels
        logger.warning("  Last 20 positions (input -> label, trainable):")
        for i in range(max(0, len(labels) - 20), len(labels)):
            input_tok = tokens[i].item()
            label_tok = labels[i].item()
            is_trainable = labels[i].item() != self.mask_token_id
            logger.warning(
                "    [%d] input=%d -> label=%d, trainable=%s",
                i,
                input_tok,
                label_tok,
                is_trainable,
            )
        logger.warning("=" * 80)

    def load_state_dict(self, state_dict):
        self._next_idx = state_dict["_next_idx"]
        self._data_stream.seek(self._next_idx + 1)

    def state_dict(self):
        return {"_next_idx": self._next_idx}


def build_finetune_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: HfTokenizerWithPadToken,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for finetuning datasets.

    Supports two data formats:
    1. "text" - Plain text for continued pretraining (all tokens trainable)
    2. "messages" - Chat/instruction format (only assistant responses trainable)

    And two data sources:
    1. "huggingface" - Load from HuggingFace Hub
    2. "local_jsonl" - Load from local JSONL file

    Args:
        dp_world_size: Data parallelism world size
        dp_rank: Data parallelism rank
        tokenizer: Tokenizer with pad token support
        job_config: Job configuration
        infinite: Whether to loop the dataset infinitely
    """
    # Get config values
    dataset_path = job_config.training.dataset
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    # Handle hf:// prefix for dataset path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]  # Remove hf:// prefix
        data_source = "huggingface"
    else:
        # Check if it's a local file
        data_source = getattr(job_config.training, "datasource", "huggingface")

    data_format = getattr(job_config.training, "dataset_format", "text")
    document_packing = getattr(job_config.training, "document_packing", True)

    # Build data stream
    if data_source == "local_jsonl":
        stream = HuggingFaceStream.from_local_jsonl(
            dataset_path=dataset_path,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
        )
    else:
        # HuggingFace dataset
        hf_token = getattr(job_config.training, "hf_token", None)
        dataset_config = job_config.training.dataset_config_name
        data_files = getattr(job_config.training, "data_files", None)
        split = job_config.training.dataset_split

        stream = HuggingFaceStream.from_hf(
            dataset_path=dataset_path,
            split=split,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            token=hf_token,
            name=dataset_config,
            data_files=data_files,
            infinite=infinite,
        )

    pad_token_id = tokenizer.pad_id

    # Build dataset based on format
    if data_format == "text":
        # Plain text format - all tokens trainable
        text_column = job_config.training.text_column

        def extract_text_column(item: dict) -> str:
            if text_column not in item:
                raise KeyError(
                    f"Text column '{text_column}' not found. "
                    f"Available: {list(item.keys())}"
                )
            return item[text_column]

        text_stream = stream.map(extract_text_column).map(tokenize_no_mask(tokenizer))

        if document_packing:
            dataset = PackedSequencesDataset(
                data_stream=text_stream,
                seq_len=seq_len,
                split_transcripts=True,
                pad_token_id=pad_token_id,
            )
        else:
            dataset = SingleSequenceDataset(
                data_stream=text_stream,
                seq_len=seq_len,
                pad_token_id=pad_token_id,
            )

    elif data_format == "messages":
        # Chat/instruction format - only assistant responses trainable
        # Load HuggingFace tokenizer for chat template
        try:
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(job_config.model.hf_assets_path)
        except ImportError:
            raise ImportError(
                "transformers package required for messages format. "
                "Install with: pip install transformers"
            )

        # Get chat template config
        start_seq = getattr(
            job_config.training,
            "chat_start_sequence",
            "<|im_start|>assistant\n",
        )
        end_seq = getattr(
            job_config.training,
            "chat_end_sequence",
            "<|im_end|>",
        )

        chat_template = ChatTemplate.from_hf_tokenizer(
            tokenizer=hf_tokenizer,
            start_of_generation_sequence=start_seq,
            end_of_generation_sequence=end_seq,
        )

        messages_stream = (
            stream.map(MessageRow.from_dict)
            .map(chat_template.format)
            .map(tokenize_and_mask(tokenizer, chat_template))
        )

        if document_packing:
            dataset = PackedSequencesDataset(
                data_stream=messages_stream,
                seq_len=seq_len,
                split_transcripts=False,  # Don't split conversations
                pad_token_id=pad_token_id,
            )
        else:
            dataset = SingleSequenceDataset(
                data_stream=messages_stream,
                seq_len=seq_len,
                pad_token_id=pad_token_id,
            )
    else:
        raise ValueError(
            f"Invalid dataset_format: {data_format}. "
            f"Valid formats are: 'text', 'messages'"
        )

    # Build dataloader
    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_finetune_tokenizer(job_config: JobConfig) -> HfTokenizerWithPadToken:
    """Build a tokenizer with pad token support for finetuning.

    Args:
        job_config: Job configuration containing model.hf_assets_path

    Returns:
        HfTokenizerWithPadToken instance
    """
    return HfTokenizerWithPadToken(job_config.model.hf_assets_path)
