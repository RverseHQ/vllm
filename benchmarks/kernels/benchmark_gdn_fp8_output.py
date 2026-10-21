# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and check the production GDN CU without importing the vLLM package.

CUDA_HOME=... TORCH_CUDA_ARCH_LIST=8.9 .venv/bin/python \
    benchmarks/kernels/benchmark_gdn_fp8_output.py --build-dir runs/gdn-sm89
For an offline SM90 build, set TORCH_CUDA_ARCH_LIST=9.0 and add --build-only.
The GDN CU uses its production --use_fast_math flag; the independent official
quantizer CU is compiled separately without that flag, as in the main build.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]

_GDN_BINDING = r"""
#include "ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(qwen_gdn_probe, m) {
  m.def("gdn(Tensor qkv, Tensor a, Tensor b, Tensor a_log, Tensor bias, "
        "Tensor idx, Tensor cu, Tensor accepted, Tensor! state, Tensor gate, "
        "Tensor norm, Tensor! out, float scale, float eps, str activation) -> ()");
  m.def("fp8(Tensor qkv, Tensor a, Tensor b, Tensor a_log, Tensor bias, "
        "Tensor idx, Tensor cu, Tensor accepted, Tensor! state, Tensor gate, "
        "Tensor norm, Tensor! out, Tensor! scales, float scale, float eps, "
        "str activation) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(qwen_gdn_probe, CUDA, m) {
  m.impl("gdn", TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
  m.impl("fp8", TORCH_BOX(&fused_gdn_decode_post_conv_mtp_fp8));
}
"""
_QUANT_BINDING = r"""
#include "ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(qwen_gdn_probe, m) {
  m.def("quant(Tensor input, Tensor! q, Tensor! scales, int group, float eps, "
        "float minimum, float maximum, bool ue8m0, bool colmajor, "
        "bool aligned) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(qwen_gdn_probe, CUDA, m) {
  m.impl("quant", TORCH_BOX(&per_token_group_quant_fp8));
}
"""


def build_probe(build_dir: Path) -> None:
    """Compile actual production source files, never copied kernel formulas."""
    includes = [str(ROOT / "csrc")]
    for name, binding, source, flags in (
        (
            "gdn",
            _GDN_BINDING,
            "csrc/gdn/fused_gdn_decode_kernel.cu",
            ["--use_fast_math"],
        ),
        (
            "quant",
            _QUANT_BINDING,
            "csrc/quantization/w8a8/fp8/per_token_group_quant.cu",
            [],
        ),
    ):
        directory = build_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        cpp = directory / "bindings.cpp"
        cpp.write_text(binding)
        load(
            name=f"qwen_gdn_fp8_probe_{name}",
            sources=[str(cpp), str(ROOT / source)],
            extra_include_paths=includes,
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-DUSE_CUDA",
                "-DVLLM_ENABLE_FUSED_GDN_DECODE",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-DUSE_CUDA",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
                "-DVLLM_ENABLE_FUSED_GDN_DECODE",
                *flags,
            ],
            # Keep symbols from an optional prebuilt reference library separate.
            extra_ldflags=["-Wl,-Bsymbolic"],
            build_directory=str(directory),
            is_python_module=False,
            verbose=True,
        )


def assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        atol=0,
        rtol=0,
    )


@torch.inference_mode()
def check_native_output(
    rows: int,
    padding: int,
    activation: str,
    norm_dtype: torch.dtype,
    *,
    installed: bool = False,
    check_reference: bool = False,
    query_lens: list[int] | None = None,
) -> None:
    """Compare to the old native BF16 output plus official native quantizer."""
    namespace = torch.ops._C if installed else torch.ops.qwen_gdn_probe
    old_op = namespace.fused_gdn_decode_post_conv_mtp if installed else namespace.gdn
    fp8_op = (
        namespace.fused_gdn_decode_post_conv_mtp_fp8 if installed else namespace.fp8
    )
    quant_op = (
        torch.ops._C.per_token_group_fp8_quant
        if installed or check_reference
        else namespace.quant
    )
    torch.manual_seed(71)
    h, hv, dim, mtp = 1, 8, 128, 4
    actual_rows = rows - padding
    ragged = query_lens is not None
    if query_lens is None:
        assert actual_rows % mtp == 0
        query_lens = [mtp] * (actual_rows // mtp)
    assert sum(query_lens) == actual_rows and all(0 <= n <= mtp for n in query_lens)
    requests = len(query_lens)
    device = "cuda"
    qkv = (
        torch.randn(
            actual_rows, (2 * h + hv) * dim, dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    a = torch.randn(actual_rows, hv, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn_like(a)
    a_log = torch.randn(hv, device=device) * 0.1
    bias = torch.randn(hv, device=device) * 0.1
    gate = torch.randn(actual_rows, hv, dim, dtype=torch.bfloat16, device=device)
    norm = torch.randn(dim, dtype=norm_dtype, device=device)
    indices = torch.arange(
        1, requests * mtp + 1, device=device, dtype=torch.int32
    ).reshape(requests, mtp)
    # A padded request with no source must emit official quantized zero rows.
    indices[-1].zero_()
    cu = torch.tensor([0, *query_lens], dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    accepted = torch.arange(requests, dtype=torch.int32, device=device) % mtp + 1
    seed = torch.randn(requests * mtp + 1, hv, dim, dim, device=device) * 0.01
    reference_state, fused_state = seed.clone(), seed.clone()
    reference = torch.zeros(rows, hv, dim, dtype=torch.bfloat16, device=device)
    fused = torch.zeros(rows, hv, dim, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.full((hv, rows), 2.0**-33, device=device).t()
    args = (qkv, a, b, a_log, bias, indices, cu, accepted)
    old_op(
        *args,
        reference_state,
        gate,
        norm,
        reference[:actual_rows],
        dim**-0.5,
        1e-6,
        activation,
    )
    if check_reference:
        sdk_state = seed.clone()
        sdk_output = torch.zeros_like(reference)
        torch.ops._C.fused_gdn_decode_post_conv_mtp(
            *args,
            sdk_state,
            gate,
            norm,
            sdk_output[:actual_rows],
            dim**-0.5,
            1e-6,
            activation,
        )
        assert_bytes_equal(reference, sdk_output)
        assert_bytes_equal(reference_state, sdk_state)
    fp8_op(
        *args,
        fused_state,
        gate,
        norm,
        fused[:actual_rows],
        scales[:actual_rows],
        dim**-0.5,
        1e-6,
        activation,
    )
    expected_q = torch.empty_like(fused).flatten(1)
    expected_scales = torch.empty(hv, rows, device=device).t()
    quant_op(
        reference.flatten(1),
        expected_q,
        expected_scales,
        128,
        1e-10,
        -448.0,
        448.0,
        True,
        True,
        False,
    )
    assert_bytes_equal(fused.flatten(1), expected_q)
    assert_bytes_equal(scales, expected_scales)
    assert_bytes_equal(fused_state, reference_state)
    # A consumer check executable on SM89; production DeepGEMM is tested on SM90
    # by the model-path suite, not replaced by an alternate production backend.
    weight = torch.randn(hv * dim, 128, dtype=torch.bfloat16, device=device)
    expected_input = expected_q.float() * expected_scales.repeat_interleave(dim, 1)
    fused_input = fused.flatten(1).float() * scales.repeat_interleave(dim, 1)
    expected_projection = expected_input.to(torch.bfloat16) @ weight
    fused_projection = fused_input.to(torch.bfloat16) @ weight
    assert_bytes_equal(fused_projection, expected_projection)
    torch.accelerator.synchronize()
    print(
        f"PASS rows={rows} padding={padding} varlen={ragged} "
        f"{activation} norm={norm_dtype}: "
        "FP8/scales/FP32-state/projection byte-identical"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--reference-library", type=Path)
    args = parser.parse_args()
    if args.reference_library is not None:
        torch.ops.load_library(str(args.reference_library.resolve()))
    build_probe(args.build_dir.resolve())
    if args.build_only:
        return
    for rows in (32, 192):
        for padding in (0, 4):
            for activation in ("silu", "sigmoid"):
                for norm_dtype in (torch.bfloat16, torch.float32):
                    check_native_output(
                        rows,
                        padding,
                        activation,
                        norm_dtype,
                        check_reference=args.reference_library is not None,
                    )
    check_native_output(
        36,
        4,
        "silu",
        torch.float32,
        query_lens=[1, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 0, 0],
        check_reference=args.reference_library is not None,
    )


if __name__ == "__main__":
    main()
