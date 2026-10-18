# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-translation-unit SM90 recovery extension; no installed SDK changes.

Precompile outside graph capture with:
    MAX_JOBS=1 python -m vllm.model_executor.layers.mamba.ops.gdn_recover_native
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_native_recovery() -> None:
    from torch.utils.cpp_extension import load

    load(
        name="gdn_recover_state_only_sm90",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cflags=["-O3"],
        # Match the producer's --use_fast_math; compile only the target H800 ISA.
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_90,code=sm_90",
        ],
        is_python_module=False,
    )


if __name__ == "__main__":
    load_native_recovery()
    print("SM90 state-only GDN recovery extension loaded")
