# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import threading

import torch

from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def maybe_launch_debugger() -> None:
    """Launch debugpy debugger if DEBUG or DEBUGPY environment variable is set.
    
    Environment variables:
        DEBUG or DEBUGPY: Set to "1", "true", or "yes" to enable debugging
        DEBUG_WAIT_RANKS: Comma-separated list of ranks to wait for debugger, 
                          or "all" (default) to wait on all ranks
        DEBUG_TIMEOUT: Timeout in seconds to wait for debugger attachment (default: 30)
    """
    device_module, device_type = utils.device_module, utils.device_type
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    # pyrefly: ignore [missing-attribute]
    device_module.set_device(device)

    # Enable debugpy if DEBUG or DEBUGPY environment variable is set
    debug = os.environ.get("DEBUG", os.environ.get("DEBUGPY", "0"))
    if debug.lower() in ("1", "true", "yes"):
        try:
            # pyrefly: ignore [missing-import]
            import debugpy

            # Use LOCAL_RANK for port assignment (single node debugging)
            # For multi-node, you'd want to use RANK and adjust ports accordingly
            port = 5678 + local_rank
            debugpy.listen(("127.0.0.1", port))
            logger.info(f"Rank {local_rank} debugger listening on port {port}")

            # Optionally wait for debugger attachment
            # Set DEBUG_WAIT_RANKS="0,1,2" to wait for specific ranks, or "all" for all ranks
            # Default is "all" to wait for all ranks
            wait_ranks_str = os.environ.get("DEBUG_WAIT_RANKS", "all")
            should_wait = False

            if wait_ranks_str.lower() == "all":
                should_wait = True
            elif wait_ranks_str:
                wait_ranks = [int(r.strip()) for r in wait_ranks_str.split(",")]
                should_wait = local_rank in wait_ranks

            if should_wait:
                logger.info(
                    f"Rank {local_rank} waiting for debugger attach on port {port}..."
                )

                # Wait for debugger with timeout (default 30 seconds, configurable via DEBUG_TIMEOUT)
                timeout_str = os.environ.get("DEBUG_TIMEOUT", "30")
                timeout = float(timeout_str) if timeout_str else 30.0
                debugger_attached = threading.Event()

                def wait_for_debugger():
                    try:
                        debugpy.wait_for_client()
                        debugger_attached.set()
                    except Exception:
                        pass

                wait_thread = threading.Thread(target=wait_for_debugger, daemon=True)
                wait_thread.start()

                if debugger_attached.wait(timeout=timeout):
                    logger.info(f"Rank {local_rank} debugger attached!")
                else:
                    logger.info(
                        f"Rank {local_rank} debugger not attached within {timeout}s, continuing..."
                    )
            else:
                logger.info(
                    f"Rank {local_rank} will not wait for debugger (use DEBUG_WAIT_RANKS to change)"
                )
        except ImportError:
            logger.error(
                "DEBUG/DEBUGPY is set but debugpy is not installed. "
                "Install it with: pip install debugpy"
            )
