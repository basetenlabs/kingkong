# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import re
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.tools.logging import logger

from .model import BaseModelArgs

# Files to copy from source HF assets to checkpoint directory
HF_ASSET_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # For SentencePiece tokenizers
    "vocab.json",
    "merges.txt",
    "*.py",  # Custom model/config Python files (e.g., configuration_nemotron_h.py)
]


class BaseStateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    Args:
        model_args: for initializing the model's memory space
        hf_assets_path: path to HF assets folder containing tokenizer, model weights, etc.
    """

    fqn_to_index_mapping: Dict[Any, int] | None
    hf_assets_path: str | None

    @abstractmethod
    def __init__(
        self,
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
        pass

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict

        Returns:
            The converted native model state dict
        """
        pass

    @abstractmethod
    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """Returns hf storage reader to read HF checkpoint

        Args:
            path: the path to read HF checkpoint

        Returns:
            The HuggingFace storage reader to read from HF checkpoint

        """
        pass

    def copy_hf_assets_to_checkpoint(self, checkpoint_dir: str) -> None:
        """Copy HF metadata files (config, tokenizer, etc.) to checkpoint directory.

        Also generates the model.safetensors.index.json file based on fqn_to_index_mapping.

        Args:
            checkpoint_dir: The directory where the HF checkpoint is being saved
        """
        if not self.hf_assets_path:
            logger.warning(
                "hf_assets_path not set, skipping copy of HF config/tokenizer files to checkpoint"
            )
            return

        # Copy config and tokenizer files
        for pattern in HF_ASSET_PATTERNS:
            src_pattern = os.path.join(self.hf_assets_path, pattern)
            # Handle glob patterns (e.g., *.py)
            if "*" in pattern:
                matching_files = glob.glob(src_pattern)
                for src_path in matching_files:
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(checkpoint_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {filename} to checkpoint directory")
            elif os.path.exists(src_pattern):
                dst_path = os.path.join(checkpoint_dir, pattern)
                shutil.copy2(src_pattern, dst_path)
                logger.info(f"Copied {pattern} to checkpoint directory")

        # Generate model.safetensors.index.json
        if self.fqn_to_index_mapping:
            self._generate_safetensors_index(checkpoint_dir)

    def _generate_safetensors_index(self, checkpoint_dir: str) -> None:
        """Generate model.safetensors.index.json from fqn_to_index_mapping.

        Args:
            checkpoint_dir: The directory where the HF checkpoint is being saved
        """
        if not self.fqn_to_index_mapping:
            return

        # Build weight_map: tensor_name -> filename
        weight_map = {}
        unique_indices = set(self.fqn_to_index_mapping.values())
        num_shards = max(unique_indices)

        for fqn, idx in self.fqn_to_index_mapping.items():
            # Format: model-00001-of-00013.safetensors
            filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            weight_map[fqn] = filename

        # Calculate total size by scanning the safetensor files
        total_size = 0
        for idx in unique_indices:
            filename = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }

        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Generated model.safetensors.index.json with {len(weight_map)} entries")


class StateDictAdapter(BaseStateDictAdapter):
    """State dict adapter base class which provides convenient default behavior to build fqn_to_index_mapping"""

    def __init__(
        self,
        model_args: BaseModelArgs,
        hf_assets_path: str | None,
    ):
        self.hf_assets_path = hf_assets_path

        if hf_assets_path:
            mapping_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
            try:
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
            except FileNotFoundError:
                logger.warning(
                    f"model.safetensors.index.json not found at hf_assets_path: {mapping_path}. \
                    Defaulting to saving a single safetensors file if checkpoint is saved in HF format"
                )
                hf_safetensors_indx = None

            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {}
                for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                    # pyrefly: ignore [missing-attribute]
                    indx = re.search(r"\d+", raw_indx).group(0)
                    self.fqn_to_index_mapping[hf_key] = int(indx)
            else:
                self.fqn_to_index_mapping = None
        else:
            self.fqn_to_index_mapping = None

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            logger.warning(
                "Loading from quantized checkpoint format is not supported for this model."
            )
        return HuggingFaceStorageReader(path)
