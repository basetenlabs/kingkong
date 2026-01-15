# Nemotron3 (NemotronH) Hybrid Model

NVIDIA's **Nemotron3** (also known as **NemotronH**) is a hybrid architecture that combines Mamba2 state-space models with traditional Transformer attention layers and Mixture of Experts (MoE) for efficient large-scale language modeling.

## Architecture Overview

The model uses a configurable layer pattern defined by `hybrid_override_pattern`:

| Symbol | Layer Type | Description |
|--------|------------|-------------|
| `M` | Mamba2 | State-space model layer for efficient long-range sequence modeling |
| `*` | Attention | Multi-head attention with Grouped Query Attention (GQA) |
| `E` | MLP | Standard feed-forward layer |
| `O` | MoE | Mixture of Experts (128 experts, 6 active per token) |

### nano-30B Configuration

- **Total Parameters**: ~31B
- **Active Parameters**: ~3B per token (due to MoE sparse activation)
- **Layers**: 52
- **Hidden Size**: 2688
- **Max Sequence Length**: 262,144 tokens

## Quick Start

### 1. Download Model Assets

```bash
# Download tokenizer, config, and index only (fast, small files)
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir ./assets/hf \
    --assets tokenizer config index

# Or download EVERYTHING including weights (~60GB)
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir ./assets/hf \
    --all
```

### 2. Run Training

**Multi-GPU Training (8 GPUs):**
```bash
NGPU=8 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B.toml" ./run_train.sh
```

**Single GPU (debug model):**
```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/debug_model.toml" ./run_train.sh
```

## Available Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `nemotron3-nano-30B.toml` | Full Nemotron3 Nano-30B model with bf16 | Pretraining |
| `nemotron3-nano-30B-sft.toml` | Supervised Fine-Tuning config | Instruction tuning on chat data |
| `nemotron3-nano-30B-cpt.toml` | Continued Pre-Training config | Domain adaptation on text corpora |
| `debug_model.toml` | Small 16-layer model | Testing & debugging |

## Finetuning

This implementation supports two finetuning modes:

### Supervised Fine-Tuning (SFT)

Train only on assistant responses while masking user messages. Perfect for instruction tuning.

```bash
NGPU=8 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B-sft.toml" ./run_train.sh
```

**Dataset Format** (messages format):
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

Key config options for SFT:
```toml
[training]
dataset_format = "messages"           # Use chat format
document_packing = true               # Pack multiple conversations
chat_start_sequence = "<|im_start|>assistant\n"  # Start of assistant turn
chat_end_sequence = "<|im_end|>"      # End of assistant turn
```

### Continued Pre-Training (CPT)

Train on all tokens in a text corpus. Perfect for domain adaptation.

```bash
NGPU=8 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B-cpt.toml" ./run_train.sh
```

**Dataset Format** (text format):
```json
{"text": "Your domain-specific text goes here..."}
```

Key config options for CPT:
```toml
[training]
dataset_format = "text"               # Use text format
document_packing = true               # Pack documents for efficiency
text_column = "text"                  # Column containing text
```

### Custom Datasets

You can use any HuggingFace dataset or local JSONL file:

```toml
[training]
# HuggingFace dataset
dataset = "hf://your-org/your-dataset"
datasource = "huggingface"

# Or local JSONL file
dataset = "/path/to/your/data.jsonl"
datasource = "local_jsonl"
```

## Key Training Options

Edit the `.toml` config file to customize:

```toml
[training]
local_batch_size = 1      # Per-GPU batch size
seq_len = 4096            # Sequence length
dtype = "bfloat16"        # Training precision

[parallelism]
data_parallel_shard_degree = -1  # FSDP sharding (-1 = auto)
tensor_parallel_degree = 1       # Tensor parallelism

[checkpoint]
initial_load_path = "./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
initial_load_in_hf = true        # Load from HuggingFace format
```

## Hardware Requirements

- **nano-30B**: Requires multi-GPU setup with sufficient VRAM (tested on 8x B200)
- **debug_model**: Can run on a single GPU for development

## Roadmap

Currently using **FSDP** (Fully Sharded Data Parallel) to get training up and running. The following parallelism strategies are planned for future support:

| Feature | Status | Notes |
|---------|--------|-------|
| FSDP | ✅ Supported | Currently used for distributed training |
| Tensor Parallelism | 🚧 To be added | Likely needed for larger model variants |
| Context Parallelism | 🚧 To be added | Needed for very long sequence lengths |
| Expert Parallelism | 🚧 To be added | Essential for scaling MoE layers |
| MFU Optimizations | 🚧 To be added | Kernel fusion, better memory layout, etc. |

These advanced parallelism strategies will become necessary once NVIDIA releases larger models in the Nemotron series.

## Implementation Notes

The model implementation in `model/model.py` is adapted from NVIDIA's official HuggingFace implementation:
[`modeling_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)

## References

- [NVIDIA Nemotron-3-Nano-30B on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
