#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the native Q4 M64/N128 split kernel without replacing the SDK library."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((HERE / "fa3_source_manifest.json").read_text())
    space = args.source_dir / "build" / "vllm-fa3-fp8" / "q4-m64n128"
    source, build = space / "source", space / "build"
    for directory in (source, build, space / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    for relative, expected in pin["git_blobs"].items():
        data = (args.source_dir / relative).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unpinned source: {relative}")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source_dir / relative, target)
    subprocess.run(
        [
            "patch",
            "--batch",
            "--fuzz=0",
            "-p1",
            "-i",
            str(HERE / "fa3_fp8_pv_sliced.patch"),
        ],
        cwd=source,
        check=True,
    )
    tile = source / "hopper/tile_size.h"
    text = tile.read_text()
    anchor = "    if (element_size == 2) {"
    assert text.count(anchor) == 2
    tile.write_text(
        text.replace(
            anchor,
            "    if (element_size == 1 && headdim == 256 && headdim_v == 256 && "
            "paged_kv_non_TMA) {\n        return {64, 128, true, false};\n    }\n"
            + anchor,
            1,
        )
    )
    api = source / "hopper/flash_api.cpp"
    text = api.read_text()
    start = text.index(
        "void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {"
    )
    end = text.index("\nvoid run_mha_fwd_combine(", start)
    dispatch = (
        "void run_q4_m64(Flash_fwd_params&, cudaStream_t);\n"
        "void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {\n"
        "    STD_TORCH_CHECK(params.arch == 90 && params.is_e4m3 && "
        "params.d == 256 && params.dv == 256);\n"
        "    STD_TORCH_CHECK(params.seqlen_q == 4 && params.h == 24 && "
        "params.h_k == 4);\n"
        "    STD_TORCH_CHECK(params.page_table && !params.pagedkv_tma && "
        "params.num_splits > 1);\n"
        "    STD_TORCH_CHECK(params.is_causal && !params.is_local && "
        "params.softcap == 0.0);\n"
        "    STD_TORCH_CHECK(params.v_dim_stride == 1 && params.cu_seqlens_q && "
        "!params.knew_ptr && !params.qv_ptr);\n"
        "    run_q4_m64(params, stream);\n"
        "}\n"
    )
    api.write_text(text[:start] + dispatch + text[end:])
    kernel = source / "hopper/flash_q4_m64.cu"
    kernel.write_text("""#include "flash_fwd_launch_template.h"
void run_q4_m64(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<90, 256, 256, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t,
        true, false, false, true, true, false, false, false, true, true,
        false, false, 1>(params, stream);
}
""")
    prefix, name = "qwen_q4_m64", "_qwen_q4_m64_C"
    symbols = (
        "Qkv_params",
        "Flash_fwd_params",
        "Flash_bwd_params",
        "run_mha_fwd_",
        "run_mha_fwd",
        "run_flash_fwd",
        "run_mha_fwd_combine_",
        "run_mha_fwd_combine",
        "run_flash_fwd_combine",
        "prepare_varlen_num_blocks",
        "mha_fwd",
        "mha_fwd_get_scheduler_metadata",
        "set_params_fprop",
        "set_params_dgrad",
        "mha_combine",
        "mha_bwd",
        "run_q4_m64",
    )
    defines = [
        "FLASHATTENTION_DISABLE_BACKWARD",
        "FLASHATTENTION_DISABLE_DROPOUT",
        "FLASHATTENTION_DISABLE_UNEVEN_K",
        "FLASHATTENTION_DISABLE_PYBIND",
        "FLASHATTENTION_VARLEN_ONLY",
        "FLASHATTENTION_PACKGQA_ONLY",
        "FLASHATTENTION_DISABLE_CLUSTER",
        "FLASHATTENTION_DISABLE_SM8x",
        "FLASHATTENTION_DISABLE_APPENDKV",
        "CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "CUTLASS_ENABLE_GDC_FOR_SM90",
        "CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        "TORCH_TARGET_VERSION=0x020A000000000000ULL",
        "USE_CUDA",
        f"flash={prefix}",
        *(f"{symbol}={prefix}_{symbol}" for symbol in symbols),
    ]
    common = ["-O3", "-DNDEBUG", "-std=c++20", *("-D" + value for value in defines)]
    cuda = [
        *common,
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-Xcompiler=-fvisibility=hidden",
        "--threads=1",
    ]
    os.environ.update(
        TMPDIR=str(space / "tmp"), MAX_JOBS="4", TORCH_CUDA_ARCH_LIST="9.0a"
    )
    from torch.utils.cpp_extension import load

    sources = [
        source / "hopper" / file
        for file in (
            "flash_api.cpp",
            "flash_api_torch_lib.cpp",
            "flash_fwd_combine.cu",
            "flash_prepare_scheduler.cu",
            "flash_q4_m64.cu",
        )
    ]
    library = load(
        name=name,
        sources=[str(p) for p in sources],
        extra_cflags=[*common, "-fvisibility=hidden"],
        extra_cuda_cflags=cuda,
        extra_ldflags=["-Wl,-Bsymbolic", "-lcuda"],
        extra_include_paths=[
            str(source / "hopper"),
            str(source / "csrc/common"),
            str(args.cutlass_dir / "include"),
        ],
        build_directory=str(build),
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    manifest = {
        "library": str(library),
        "sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        "namespace": name,
        "cflags": common,
        "cuda_flags": cuda,
        "source_sha256": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
    }
    (space / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
