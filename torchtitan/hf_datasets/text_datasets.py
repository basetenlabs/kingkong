# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict
from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


HF_DATASET_PREFIX = "hf://"


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _create_custom_hf_loader(
    split: str = "train",
    config_name: str | None = None,
) -> Callable:
    """Create a loader function for custom HuggingFace datasets."""

    def loader(dataset_path: str):
        logger.info(
            f"Loading custom HuggingFace dataset from {dataset_path} "
            f"(split={split}, config={config_name})"
        )
        return load_dataset(
            dataset_path,
            name=config_name,
            split=split,
            streaming=True,
        )

    return loader


def _create_custom_text_processor(text_column: str = "text") -> Callable:
    """Create a text processor for custom datasets with configurable column name."""

    def processor(sample: dict[str, Any]) -> str:
        if text_column not in sample:
            available_columns = list(sample.keys())
            raise KeyError(
                f"Text column '{text_column}' not found in dataset sample. "
                f"Available columns: {available_columns}. "
                f"Set 'text_column' in your config to the correct column name."
            )
        return sample[text_column]

    return processor


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str,
    dataset_path: str | None = None,
    text_column: str = "text",
    dataset_split: str = "train",
    dataset_config_name: str | None = None,
) -> tuple[str, Callable, Callable]:
    """
    Validate dataset name and path.

    Supports two modes:
    1. Registered datasets: Use dataset_name to look up pre-configured datasets
    2. Custom HF datasets: Use "hf://<path>" prefix to load any HuggingFace dataset

    Args:
        dataset_name: Name of registered dataset or "hf://<hf_path>" for custom datasets
        dataset_path: Optional override path for the dataset
        text_column: Column name containing text (for custom datasets)
        dataset_split: Split to load (for custom datasets)
        dataset_config_name: Optional config/subset name (for custom datasets)

    Returns:
        Tuple of (path, loader_fn, processor_fn)
    """
    # Check if this is a custom HuggingFace dataset (hf:// prefix)
    if dataset_name.startswith(HF_DATASET_PREFIX):
        hf_path = dataset_name[len(HF_DATASET_PREFIX) :]
        path = dataset_path or hf_path
        loader = _create_custom_hf_loader(
            split=dataset_split,
            config_name=dataset_config_name,
        )
        processor = _create_custom_text_processor(text_column)
        logger.info(
            f"Preparing custom HuggingFace dataset '{hf_path}' from {path} "
            f"(split={dataset_split}, text_column={text_column})"
        )
        return path, loader, processor

    # Otherwise, use registered datasets
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}. "
            f"To load a custom HuggingFace dataset, use the 'hf://' prefix, "
            f"e.g., 'hf://allenai/c4' or 'hf://username/dataset_name'."
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        text_column: str = "text",
        dataset_split: str = "train",
        dataset_config_name: str | None = None,
    ) -> None:
        # Force lowercase for consistent comparison (but preserve hf:// prefix case for paths)
        if not dataset_name.startswith(HF_DATASET_PREFIX):
            dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name,
            dataset_path,
            text_column=text_column,
            dataset_split=dataset_split,
            dataset_config_name=dataset_config_name,
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    text_column = job_config.training.text_column
    dataset_split = job_config.training.dataset_split
    dataset_config_name = job_config.training.dataset_config_name

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        text_column=text_column,
        dataset_split=dataset_split,
        dataset_config_name=dataset_config_name,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len
    text_column = job_config.validation.text_column
    dataset_split = job_config.validation.dataset_split
    dataset_config_name = job_config.validation.dataset_config_name

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        text_column=text_column,
        dataset_split=dataset_split,
        dataset_config_name=dataset_config_name,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
