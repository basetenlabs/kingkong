# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert DCP checkpoints to HuggingFace format.

For LoRA fine-tuned checkpoints:
  - Default (--adapters-only): Export LoRA adapters in PEFT format for serving with vLLM
  - With --merge-loras: Merge LoRA weights into base model weights

Usage:
    # Standard conversion (non-LoRA checkpoint)
    python convert_to_hf.py <input_dir> <output_dir> --model_name llama3 --model_flavor 8B

    # LoRA checkpoint - export adapters only (default for LoRA)
    python convert_to_hf.py <input_dir> <output_dir> \\
        --model_name deepseek_v3 --model_flavor deepseek_aghilora \\
        --base_model_name_or_path Aghilan/dvs3.1-fugazzi

    # LoRA checkpoint - merge into base weights
    python convert_to_hf.py <input_dir> <output_dir> \\
        --model_name deepseek_v3 --model_flavor deepseek_aghilora \\
        --merge-loras
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from safetensors.torch import save_file
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.tools.logging import init_logger, logger


# =============================================================================
# HuggingFace Assets
# =============================================================================

def copy_hf_assets(hf_assets_path, output_dir):
    """
    Copy config.json and tokenizer files from hf_assets_path to output_dir.
    """
    hf_assets_path = Path(hf_assets_path)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Required files
    config_source = hf_assets_path / "config.json"
    if not config_source.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_source}. "
            f"Please ensure the HuggingFace assets path is correct."
        )
    shutil.copy2(config_source, output_dir / "config.json")
    
    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    for fname in tokenizer_files:
        src = hf_assets_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)


# =============================================================================
# LoRA Detection and Utilities
# =============================================================================

def has_lora_weights(state_dict):
    """Check if the state dict contains LoRA weights."""
    return any(".lora_A.weight" in k or ".lora_B.weight" in k for k in state_dict.keys())


def is_lora_checkpoint(model_args):
    """Check if model_args indicates a LoRA fine-tuned model."""
    return hasattr(model_args, 'finetune_lora_rank') and model_args.finetune_lora_rank > 0


# =============================================================================
# LoRA Merge (into base weights)
# =============================================================================

def merge_lora_weights(state_dict, model_args):
    """
    Merge LoRA fine-tuning weights into base weights.
    
    Merge formula: merged = base + lora_B @ lora_A * scale
    where scale = finetune_lora_alpha / finetune_lora_rank
    """
    if model_args.finetune_lora_rank <= 0:
        return state_dict
    
    scale = model_args.finetune_lora_alpha / model_args.finetune_lora_rank
    lora_a_keys = [k for k in state_dict.keys() if ".lora_A.weight" in k]
    
    if not lora_a_keys:
        raise ValueError(
            f"finetune_lora_rank={model_args.finetune_lora_rank} but no LoRA keys found. "
            "Expected keys matching pattern '*.lora_A.weight'"
        )
    
    logger.info("=" * 60)
    logger.info("[LoRA Merge] Merging LoRA weights into base model...")
    logger.info("=" * 60)
    logger.info(f"[LoRA] Scale: {scale:.4f} (alpha={model_args.finetune_lora_alpha}, rank={model_args.finetune_lora_rank})")
    
    merged_layers = []
    for lora_a_key in sorted(lora_a_keys):
        lora_module_key = lora_a_key.replace(".lora_A.weight", "")
        prefix = lora_module_key.rsplit(".", 1)[0]
        lora_module_name = lora_module_key.rsplit(".", 1)[1]
        
        if not lora_module_name.startswith("finetune_lora_"):
            logger.warning(f"[LoRA] Skipping '{lora_module_name}' - doesn't start with 'finetune_lora_'")
            continue
        
        base_relative = lora_module_name.replace("finetune_lora_", "").replace("__", ".")
        lora_b_key = f"{lora_module_key}.lora_B.weight"
        base_key = f"{prefix}.{base_relative}.weight"
        
        if lora_b_key not in state_dict:
            raise KeyError(f"Missing LoRA B key: {lora_b_key}")
        if base_key not in state_dict:
            raise KeyError(f"Missing base weight key: {base_key}")
        
        base_weight = state_dict[base_key]
        lora_a_weight = state_dict[lora_a_key]
        lora_b_weight = state_dict[lora_b_key]
        
        # Merge: merged = base + lora_B @ lora_A * scale
        delta = lora_b_weight @ lora_a_weight * scale
        state_dict[base_key] = base_weight + delta
        
        del state_dict[lora_a_key]
        del state_dict[lora_b_key]
        merged_layers.append(base_key)
        
        logger.info(f"  ✓ Merged {lora_module_key} → {base_key}")
    
    logger.info("")
    logger.info(f"[LoRA] Merged {len(merged_layers)} LoRA adapter(s) into base weights")
    return state_dict


# =============================================================================
# LoRA Adapter Extraction (PEFT format)
# =============================================================================

def convert_tt_to_hf_lora_path(prefix: str, base_relative: str) -> str:
    """Convert TorchTitan layer path to HuggingFace path for LoRA adapters."""
    parts = prefix.split(".")
    
    if len(parts) >= 3 and parts[0] == "layers":
        layer_idx = parts[1]
        module_type = parts[2]
        
        if module_type == "attention":
            proj_mapping = {
                "wo": "o_proj", "wq": "q_proj", "wk": "k_proj", "wv": "v_proj",
                "wqkv": "qkv_proj", "wq_a": "q_a_proj", "wq_b": "q_b_proj",
                "wkv_a": "kv_a_proj_with_mqa", "wkv_b": "kv_b_proj",
            }
            hf_proj = proj_mapping.get(base_relative, base_relative)
            if base_relative not in proj_mapping:
                logger.warning(f"[Path Mapping] Unknown attention projection '{base_relative}'")
            return f"model.layers.{layer_idx}.self_attn.{hf_proj}"
        
        elif module_type in ("mlp", "feed_forward"):
            proj_mapping = {
                "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
                "gate": "gate_proj", "down": "down_proj", "up": "up_proj",
            }
            hf_proj = proj_mapping.get(base_relative, base_relative)
            if base_relative not in proj_mapping:
                logger.warning(f"[Path Mapping] Unknown MLP projection '{base_relative}'")
            return f"model.layers.{layer_idx}.mlp.{hf_proj}"
        
        else:
            logger.warning(f"[Path Mapping] Unknown module type '{module_type}'")
    
    logger.warning(f"[Path Mapping] Fallback: model.{prefix}.{base_relative}")
    return f"model.{prefix}.{base_relative}"


def extract_lora_adapters(state_dict, model_args):
    """
    Extract LoRA weights from state dict in PEFT format.
    
    Returns:
        tuple: (lora_state_dict, target_modules, base_state_dict)
    """
    logger.info("=" * 60)
    logger.info("[LoRA Extraction] Extracting LoRA adapters (PEFT format)...")
    logger.info("=" * 60)
    
    if model_args.finetune_lora_rank <= 0:
        raise ValueError("finetune_lora_rank must be > 0 for adapter extraction.")
    
    logger.info(f"[LoRA] Rank: {model_args.finetune_lora_rank}, Alpha: {model_args.finetune_lora_alpha}")
    logger.info(f"[LoRA] Scaling factor: {model_args.finetune_lora_alpha / model_args.finetune_lora_rank:.4f}")
    
    all_keys = list(state_dict.keys())
    lora_a_keys = [k for k in all_keys if ".lora_A.weight" in k]
    lora_b_keys = [k for k in all_keys if ".lora_B.weight" in k]
    
    logger.info(f"[LoRA] Found {len(lora_a_keys)} lora_A keys, {len(lora_b_keys)} lora_B keys")
    
    if not lora_a_keys:
        raise ValueError("No LoRA keys found in checkpoint.")
    
    # Log all LoRA keys found
    all_lora_keys = [k for k in all_keys if "lora_" in k.lower() or "finetune_lora" in k]
    if all_lora_keys:
        logger.info("[LoRA] All LoRA-related keys:")
        for key in sorted(all_lora_keys):
            shape = tuple(state_dict[key].shape) if key in state_dict else "N/A"
            logger.info(f"  - {key} (shape: {shape})")
    
    lora_state_dict = {}
    target_modules = set()
    keys_to_remove = []
    processed_layers = []
    
    logger.info("")
    logger.info("[LoRA] Processing layers:")
    logger.info("-" * 60)
    
    for lora_a_key in sorted(lora_a_keys):
        lora_module_key = lora_a_key.replace(".lora_A.weight", "")
        prefix = lora_module_key.rsplit(".", 1)[0]
        lora_module_name = lora_module_key.rsplit(".", 1)[1]
        
        if not lora_module_name.startswith("finetune_lora_"):
            logger.warning(f"[LoRA] Skipping '{lora_module_name}' - invalid prefix")
            continue
        
        base_relative = lora_module_name.replace("finetune_lora_", "").replace("__", ".")
        lora_b_key = f"{lora_module_key}.lora_B.weight"
        
        if lora_b_key not in state_dict:
            raise KeyError(f"Missing lora_B key: {lora_b_key}")
        
        lora_a_weight = state_dict[lora_a_key]
        lora_b_weight = state_dict[lora_b_key]
        a_shape = tuple(lora_a_weight.shape)
        b_shape = tuple(lora_b_weight.shape)
        
        # Convert to PEFT format
        peft_base_path = convert_tt_to_hf_lora_path(prefix, base_relative)
        peft_a_key = f"base_model.model.{peft_base_path}.lora_A.weight"
        peft_b_key = f"base_model.model.{peft_base_path}.lora_B.weight"
        
        lora_state_dict[peft_a_key] = lora_a_weight
        lora_state_dict[peft_b_key] = lora_b_weight
        
        target_module = peft_base_path.split(".")[-1]
        target_modules.add(target_module)
        keys_to_remove.extend([lora_a_key, lora_b_key])
        
        logger.info(f"  ✓ {lora_module_key}")
        logger.info(f"      → PEFT: {peft_a_key}")
        logger.info(f"      Shapes: A={a_shape}, B={b_shape}")
        
        processed_layers.append({"target_module": target_module})
    
    base_state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("[LoRA Extraction] Summary:")
    logger.info(f"  Total LoRA pairs: {len(processed_layers)}")
    logger.info(f"  Target modules: {sorted(target_modules)}")
    
    module_counts = {}
    for layer in processed_layers:
        mod = layer["target_module"]
        module_counts[mod] = module_counts.get(mod, 0) + 1
    logger.info("[LoRA] Breakdown by module:")
    for mod, count in sorted(module_counts.items()):
        logger.info(f"    - {mod}: {count} layer(s)")
    logger.info("=" * 60)
    
    return lora_state_dict, target_modules, base_state_dict


def create_adapter_config(model_args, target_modules, base_model_name_or_path):
    """Create PEFT-compatible adapter_config.json."""
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": model_args.finetune_lora_alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": model_args.finetune_lora_rank,
        "target_modules": sorted(list(target_modules)),
        "task_type": "CAUSAL_LM",
    }


# =============================================================================
# Main Conversion Functions
# =============================================================================

@torch.inference_mode()
def convert_to_hf_merged(
    input_dir, output_dir, model_name, model_flavor, hf_assets_path, export_dtype
):
    """Convert DCP checkpoint to HF format, merging LoRA if present."""
    logger.info(f"Starting DCP to HF conversion (merge mode): {input_dir} -> {output_dir}")
    logger.info(f"Model: {model_name}/{model_flavor}, export_dtype: {export_dtype}")

    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    logger.info(f"Loaded train spec for {model_name}/{model_flavor}")

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    # pyrefly: ignore [bad-argument-type]
    model = ModelWrapper(model)
    logger.info("Created model instance on CPU")

    # pyrefly: ignore [not-callable]
    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert sd_adapter is not None, "sd_adapter is required for HF conversion"

    state_dict = model._get_state_dict()
    logger.info(f"Loading DCP checkpoint from {input_dir}...")
    dcp.load(state_dict, checkpoint_id=input_dir)
    logger.info(f"Loaded checkpoint with {len(state_dict)} keys")

    # Merge LoRA if present
    if is_lora_checkpoint(model_args):
        state_dict = merge_lora_weights(state_dict, model_args)

    # Convert to HF format
    logger.info("Converting state dict to HuggingFace format...")
    hf_state_dict = sd_adapter.to_hf(state_dict)
    logger.info(f"Converted to HF format with {len(hf_state_dict)} keys")

    # Filter training-only keys
    if sd_adapter.fqn_to_index_mapping is not None:
        hf_index_keys = set(sd_adapter.fqn_to_index_mapping.keys())
        hf_state_dict = {k: v for k, v in hf_state_dict.items() if k in hf_index_keys}
    else:
        training_only = ["expert_bias", "e_score_correction_bias"]
        hf_state_dict = {k: v for k, v in hf_state_dict.items() 
                         if not any(p in k for p in training_only)}
    logger.info(f"Filtered state dict has {len(hf_state_dict)} keys")

    # Apply export dtype
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        logger.info(f"Converting tensors to {export_dtype}...")
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # Save
    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    
    output_dir_abs = Path(output_dir).resolve()
    logger.info(f"Saving HF checkpoint to: {output_dir_abs}")
    dcp.save(hf_state_dict, storage_writer=storage_writer)
    
    copy_hf_assets(hf_assets_path, output_dir)
    logger.info(f"Conversion complete! HF checkpoint saved to: {output_dir_abs}")


@torch.inference_mode()
def convert_to_hf_adapters(
    input_dir, output_dir, model_name, model_flavor, hf_assets_path, 
    export_dtype, base_model_name_or_path
):
    """Convert DCP checkpoint to PEFT-compatible LoRA adapter format.
    
    This function is memory-optimized to load only LoRA adapter weights,
    not the full model checkpoint. This is achieved by:
    1. Creating the model on meta device (no memory allocation)
    2. Filtering the state dict to only LoRA keys
    3. Loading only those keys from the DCP checkpoint
    """
    logger.info(f"Starting DCP to PEFT LoRA conversion: {input_dir} -> {output_dir}")
    logger.info(f"Model: {model_name}/{model_flavor}, export_dtype: {export_dtype}")
    logger.info(f"Base model: {base_model_name_or_path}")

    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    logger.info(f"Loaded train spec for {model_name}/{model_flavor}")
    # pyrefly: ignore [missing-attribute]
    logger.info(f"LoRA config: rank={model_args.finetune_lora_rank}, alpha={model_args.finetune_lora_alpha}")

    # Create model on meta device - no memory allocated for weights
    with torch.device("meta"):
        model = train_spec.model_cls(model_args)
    # pyrefly: ignore [bad-argument-type]
    model = ModelWrapper(model)
    logger.info("Created model structure on meta device (no memory allocated)")

    # Get full state dict structure (meta tensors), then filter to LoRA keys only
    full_state_dict = model._get_state_dict()
    lora_keys = [k for k in full_state_dict.keys() 
                 if ".lora_A.weight" in k or ".lora_B.weight" in k]
    
    if not lora_keys:
        raise ValueError(
            f"No LoRA keys found in model structure. "
            f"Ensure model_flavor '{model_flavor}' has finetune_lora_rank > 0."
        )
    
    logger.info(f"Found {len(lora_keys)} LoRA keys in model structure")
    logger.info(f"Skipping {len(full_state_dict) - len(lora_keys)} base model keys (memory optimization)")
    
    # Allocate real CPU tensors only for LoRA keys
    lora_state_dict = {
        k: torch.empty(full_state_dict[k].shape, dtype=full_state_dict[k].dtype, device="cpu")
        for k in lora_keys
    }
    
    # Load only LoRA weights from checkpoint
    logger.info(f"Loading {len(lora_state_dict)} LoRA weights from DCP checkpoint...")
    dcp.load(lora_state_dict, checkpoint_id=input_dir)
    logger.info(f"Loaded LoRA weights successfully")

    # Extract and convert to PEFT format
    lora_state_dict, target_modules, _ = extract_lora_adapters(lora_state_dict, model_args)

    # Apply export dtype
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        logger.info(f"Converting LoRA tensors to {export_dtype}...")
        lora_state_dict = {k: v.to(target_dtype) for k, v in lora_state_dict.items()}

    # Save adapter
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights_path = output_dir / "adapter_model.safetensors"
    logger.info(f"Saving LoRA adapter weights to {adapter_weights_path}...")
    save_file(lora_state_dict, adapter_weights_path)
    logger.info(f"Saved {len(lora_state_dict)} LoRA weight tensors")

    # Save config
    adapter_config = create_adapter_config(model_args, target_modules, base_model_name_or_path)
    adapter_config_path = output_dir / "adapter_config.json"
    with open(adapter_config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    logger.info(f"Saved adapter config to {adapter_config_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("LoRA Adapter Export Complete!")
    logger.info(f"  Output: {output_dir.resolve()}")
    logger.info(f"  Base model: {base_model_name_or_path}")
    # pyrefly: ignore [missing-attribute]
    logger.info(f"  LoRA rank: {model_args.finetune_lora_rank}")
    # pyrefly: ignore [missing-attribute]
    logger.info(f"  LoRA alpha: {model_args.finetune_lora_alpha}")
    logger.info(f"  Target modules: {sorted(target_modules)}")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert DCP weights to HuggingFace format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard conversion (non-LoRA)
  python convert_to_hf.py checkpoint/ output/ --model_name llama3 --model_flavor 8B

  # LoRA checkpoint - export adapters (default for LoRA checkpoints)
  python convert_to_hf.py checkpoint/ output/ \\
      --model_name deepseek_v3 --model_flavor deepseek_aghilora \\
      --base_model_name_or_path Aghilan/dvs3.1-fugazzi

  # LoRA checkpoint - merge into base weights
  python convert_to_hf.py checkpoint/ output/ \\
      --model_name deepseek_v3 --model_flavor deepseek_aghilora \\
      --merge-loras
        """
    )
    parser.add_argument("input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--hf_assets_path", type=Path,
        help="Path to HF assets directory.",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, default="llama3")
    parser.add_argument("--model_flavor", type=str, default="8B")
    parser.add_argument(
        "--export_dtype", type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Export dtype (default: bfloat16)",
    )
    
    # LoRA handling options (mutually exclusive)
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--merge-loras", action="store_true",
        help="Merge LoRA weights into base model weights.",
    )
    lora_group.add_argument(
        "--adapters-only", action="store_true",
        help="Export LoRA adapters in PEFT format (default for LoRA checkpoints).",
    )
    
    parser.add_argument(
        "--base_model_name_or_path", type=str, default=None,
        help="Base model for LoRA adapters (required with --adapters-only or for LoRA checkpoints).",
    )
    
    args = parser.parse_args()

    try:
        # Load model args to check if it's a LoRA checkpoint
        train_spec = train_spec_module.get_train_spec(args.model_name)
        model_args = train_spec.model_args[args.model_flavor]
        is_lora = is_lora_checkpoint(model_args)
        
        if is_lora:
            # pyrefly: ignore [missing-attribute]
            logger.info(f"[Config] Detected LoRA checkpoint (rank={model_args.finetune_lora_rank})")
            
            if args.merge_loras:
                # User explicitly wants to merge
                logger.info("[Config] Mode: --merge-loras (merging LoRA into base weights)")
                convert_to_hf_merged(
                    args.input_dir, args.output_dir,
                    args.model_name, args.model_flavor,
                    args.hf_assets_path, args.export_dtype,
                )
            else:
                # Default to adapters-only for LoRA checkpoints
                if not args.adapters_only:
                    logger.info("[Config] Mode: adapters-only (default for LoRA checkpoints)")
                else:
                    logger.info("[Config] Mode: --adapters-only")
                
                if not args.base_model_name_or_path:
                    parser.error(
                        "--base_model_name_or_path is required for LoRA adapter export. "
                        "Use --merge-loras if you want to merge LoRA into base weights instead."
                    )
                
                convert_to_hf_adapters(
                    args.input_dir, args.output_dir,
                    args.model_name, args.model_flavor,
                    args.hf_assets_path, args.export_dtype,
                    args.base_model_name_or_path,
                )
        else:
            # Non-LoRA checkpoint
            if args.adapters_only:
                parser.error("--adapters-only requires a LoRA checkpoint (finetune_lora_rank > 0)")
            if args.merge_loras:
                logger.warning("[Config] --merge-loras specified but no LoRA weights in checkpoint")
            
            logger.info("[Config] Mode: standard HF conversion (non-LoRA checkpoint)")
            convert_to_hf_merged(
                args.input_dir, args.output_dir,
                args.model_name, args.model_flavor,
                args.hf_assets_path, args.export_dtype,
            )
            
    except FileNotFoundError as e:
        logger.error("=" * 60)
        logger.error(f"[FAILED] File not found: {e}")
        logger.error("=" * 60)
        raise
    except KeyError as e:
        logger.error("=" * 60)
        logger.error(f"[FAILED] Key error: {e}")
        logger.error("=" * 60)
        raise
    except ValueError as e:
        logger.error("=" * 60)
        logger.error(f"[FAILED] Value error: {e}")
        logger.error("=" * 60)
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[FAILED] {type(e).__name__}: {e}")
        logger.error("=" * 60)
        raise