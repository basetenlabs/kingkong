# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable


__all__ = [
    "DatasetConfig",
    # Chat template utilities
    "ChatTemplate",
    "HfTokenizerWithPadToken",
    "MessageRow",
    "TokensAndMask",
    "tokenize_and_mask",
    "tokenize_no_mask",
    # Finetuning datasets
    "HuggingFaceStream",
    "PackedSequencesDataset",
    "SingleSequenceDataset",
    "build_finetune_dataloader",
    "build_finetune_tokenizer",
]


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable


# Import chat template utilities
from torchtitan.hf_datasets.chat_template import (
    ChatTemplate,
    HfTokenizerWithPadToken,
    MessageRow,
    TokensAndMask,
    tokenize_and_mask,
    tokenize_no_mask,
)

# Import finetuning datasets
from torchtitan.hf_datasets.finetune_datasets import (
    HuggingFaceStream,
    PackedSequencesDataset,
    SingleSequenceDataset,
    build_finetune_dataloader,
    build_finetune_tokenizer,
)
