# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GDN recovery extension compiled for the target CUDA architectures.

Precompile outside graph capture with:
    MAX_JOBS=1 python -m vllm.model_executor.layers.mamba.ops.gdn_recover_native
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_native_recovery() -> None:
    from torch.utils.cpp_extension import load

    load(
        name="gdn_recover_state_only",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cflags=["-O3"],
        # Match the producer's arithmetic; let PyTorch select CUDA architectures.
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
    )


if __name__ == "__main__":
    load_native_recovery()
    print("Native state-only GDN recovery extension loaded")
