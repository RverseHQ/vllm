#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only b273's SM90 FP8 D256 forward kernels against the active Torch.

No installation, SDK replacement, GPU execution, or downloads are performed.
All artifacts are written below source-dir/build/vllm-fa3-fp8/<variant>.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cutlass-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--variant", choices=("sliced", "stock"), default="sliced")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = args.source_dir.resolve()
    cutlass = args.cutlass_dir.resolve()
    pin = json.loads((HERE / "fa3_source_manifest.json").read_text())
    # Tarballs have no .git: validate the actual input bytes, not a directory name.
    for relative, expected in pin["git_blobs"].items():
        path = source / relative
        data = path.read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Not the pinned b273 source: {path}: {actual} != {expected}"
            )
    cutlass_header = cutlass / "include/cutlass/cutlass.h"
    if not cutlass_header.is_file():
        raise FileNotFoundError(
            f"Missing CUTLASS {pin['cutlass_commit']}: {cutlass_header}"
        )
    if (cutlass / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != pin["cutlass_commit"]:
            raise RuntimeError(
                f"CUTLASS revision {revision} != {pin['cutlass_commit']}"
            )

    space = source / "build" / "vllm-fa3-fp8" / args.variant
    isolated = space / "source"
    build = space / "build"
    scratch = space / "tmp"
    prepared = scratch / "prepared-source"
    for directory in (isolated, build, scratch, prepared):
        directory.mkdir(parents=True, exist_ok=True)
    # Generate pristine + patched inputs away from Ninja's dependency paths.
    # Re-running only to change link flags must not invalidate the CUDA objects.
    for relative in pin["git_blobs"]:
        target = prepared / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    patch = HERE / "fa3_fp8_pv_sliced.patch"
    if args.variant == "sliced":
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)],
            cwd=prepared,
            check=True,
        )
    api = prepared / "hopper/flash_api.cpp"
    original = api.read_text()
    start = original.index(
        "void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {"
    )
    end = original.index("\nvoid run_mha_fwd_combine(", start)
    api.write_text(
        original[:start]
        + (HERE / "fa3_fp8_dispatch.cpp.inc").read_text()
        + original[end:]
    )
    for relative in pin["git_blobs"]:
        target = isolated / relative
        data = (prepared / relative).read_bytes()
        if not target.is_file() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    suffix = "" if args.variant == "sliced" else "_stock"
    name = "_qwen_main237_fa3" + suffix + "_C"
    prefix = "qwen_main237_fa3" + suffix
    # Rename the upstream flash C++/CUDA namespace AND global launch/API symbols.
    # Hidden ELF visibility and Bsymbolic also prevent SDK symbol interposition.
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
    )
    definitions = [
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
    defines = ["-D" + value for value in definitions]
    # FA3's V-transpose pipeline consumes with a Producer role. Upstream release
    # builds disable CUTLASS's debug-only role-check breakpoints via NDEBUG.
    cflags = ["-O3", "-DNDEBUG", "-std=c++20", "-fvisibility=hidden", *defines]
    cuda_flags = [
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-gencode=arch=compute_90a,code=sm_90a",
        "-Xcompiler=-fvisibility=hidden",
        "--threads=1",
        *defines,
    ]
    sources = [str(isolated / relative) for relative in pin["translation_units"]]
    manifest = {
        **pin,
        "variant": args.variant,
        "op_namespace": name,
        "cpp_cuda_namespace": prefix,
        "python": sys.executable,
        "cutlass_dir": str(cutlass),
        "cutlass_header_sha256": sha256(cutlass_header),
        "patch_sha256": sha256(patch) if args.variant == "sliced" else None,
        "dispatcher_sha256": sha256(HERE / "fa3_fp8_dispatch.cpp.inc"),
        "builder_sha256": sha256(Path(__file__)),
        "sources": sources,
        "cflags": cflags,
        "cuda_flags": cuda_flags,
        "link_flags": ["-Wl,-Bsymbolic"],
        "max_jobs": args.jobs,
        "prepared_sha256": {p: sha256(isolated / p) for p in pin["git_blobs"]},
        "status": "prepared",
    }
    manifest_path = space / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(manifest_path, flush=True)
        return

    # Set scratch/arch before cpp_extension imports; no GPU discovery is needed.
    os.environ["TMPDIR"] = str(scratch)
    os.environ["MAX_JOBS"] = str(args.jobs)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("Set CUDA_HOME to the CUDA 13 toolkit (with nvcc)")
    nvcc = Path(CUDA_HOME) / "bin/nvcc"
    # CUTLASS calls the CUDA driver directly (e.g. cuTensorMapEncodeTiled).
    # The deployment provides libcuda.so in the default system link path.
    manifest["link_flags"] += ["-lcuda"]
    manifest.update(
        {
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": str(CUDA_HOME),
            "nvcc_version": subprocess.check_output(
                [str(nvcc), "--version"], text=True
            ),
            "cxx_version": subprocess.check_output(
                [os.environ.get("CXX", "c++"), "--version"], text=True
            ),
            "status": "building",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    library = load(
        name=name,
        sources=sources,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=manifest["link_flags"],
        extra_include_paths=[
            str(isolated / "hopper"),
            str(isolated / "csrc/common"),
            str(cutlass / "include"),
        ],
        build_directory=str(build),
        with_cuda=True,
        is_python_module=False,
        verbose=True,
    )
    manifest.update(
        {
            "status": "built_and_loaded",
            "library": str(library),
            "library_sha256": sha256(Path(library)),
            "objects_sha256": {p.name: sha256(p) for p in build.glob("*.o")},
            "build_ninja_sha256": sha256(build / "build.ninja"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FA3_LIBRARY={library}\nFA3_BUILD_MANIFEST={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
