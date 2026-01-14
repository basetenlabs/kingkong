# Useful Commands

## Debugging with debugpy

Enable remote debugging for attaching VS Code or other debuggers. Each rank listens on port `5678 + LOCAL_RANK`.

```bash
# Enable debugpy for attaching debugger (ports 5678+)
DEBUG=1 ./run_train.sh

# Wait for specific ranks with custom timeout (60 seconds)
DEBUG=1 DEBUG_WAIT_RANKS="0,1" DEBUG_TIMEOUT=60 ./run_train.sh

# Debug DeepSeek V3 16B on 8 GPUs, wait only for rank 0
DEBUG=1 DEBUG_WAIT_RANKS="0" NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh

# Debug Qwen3 0.6B on 4 GPUs, wait only for rank 0
NGPU=4 DEBUG=1 DEBUG_WAIT_RANKS="0" CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml" ./run_train.sh
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` or `DEBUGPY` | Set to `1`, `true`, or `yes` to enable debugging | `0` (disabled) |
| `DEBUG_WAIT_RANKS` | Comma-separated list of ranks to wait for debugger, or `all` | `all` |
| `DEBUG_TIMEOUT` | Timeout in seconds to wait for debugger attachment | `30` |
