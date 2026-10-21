# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned FA3 build inputs and the real Q4 final-combine/DeepGEMM boundary.

Set FA3_SOURCE_DIR for the CPU source-preparation test. Hopper numerical tests
require the three QWEN_MAIN237_FA3*_LIBRARY variables, including the independent
Q4_GATE_QUANT library built with build_fp8_q4.py --gate-quant.
"""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_gate_quant_source_preparation_is_reproducible(tmp_path):
    """Both patches must apply to the pinned source, without modifying it."""
    if "FA3_SOURCE_DIR" not in os.environ:
        pytest.skip("set FA3_SOURCE_DIR to the pinned b273 FA3 checkout")
    source = os.environ["FA3_SOURCE_DIR"]
    pin = json.loads(
        (ROOT / "tools/flash_attn_fp8/fa3_source_manifest.json").read_text()
    )
    before = {
        name: hashlib.sha256((Path(source) / name).read_bytes()).hexdigest()
        for name in pin["git_blobs"]
    }
    manifests = []
    for name in ("first", "second", "baseline"):
        output = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/flash_attn_fp8/build_fp8_q4.py"),
                "--source-dir",
                source,
                "--cutlass-dir",
                str(Path(source) / "csrc/cutlass"),
                "--output-dir",
                str(output),
                *(["--gate-quant"] if name != "baseline" else []),
                "--prepare-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((output / "source-manifest.json").read_text())
        expected_namespace = (
            "_qwen_q4_m64_C" if name == "baseline" else "_qwen_q4_gate_quant_C"
        )
        assert manifest["namespace"] == expected_namespace
        for relative, digest in manifest["source_sha256"].items():
            assert (
                hashlib.sha256((output / "source" / relative).read_bytes()).hexdigest()
                == digest
            )
        manifests.append(manifest)
    assert manifests[0] == manifests[1]
    fused_source = manifests[0]["source_sha256"]
    baseline_source = manifests[2]["source_sha256"]
    for protected in (
        "hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp",
        "hopper/tile_size.h",
        "hopper/flash_q4_m64.cu",
    ):
        assert fused_source[protected] == baseline_source[protected]
    for patched in (
        "hopper/flash.h",
        "hopper/flash_api.cpp",
        "hopper/flash_api_torch_lib.cpp",
        "hopper/flash_fwd_combine_kernel.h",
        "hopper/flash_fwd_combine_launch_template.h",
    ):
        assert fused_source[patched] != baseline_source[patched]
    assert before == {
        name: hashlib.sha256((Path(source) / name).read_bytes()).hexdigest()
        for name in pin["git_blobs"]
    }


@pytest.fixture
def hopper():
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90; the Q4 attention kernel is not an SM89 kernel")
    from vllm.vllm_flash_attn import flash_attn_interface as fa

    assert fa._FP8_TWO_LEVEL_FWD is not None, "load the pinned two-level FP8 library"
    assert fa._FP8_Q4_FWD is not None, "load the original pinned Q4 library"
    assert fa._FP8_Q4_GATE_QUANT_FWD is not None, "load the independent gate-quant ABI"
    return torch, fa


def _captured_gate_product(attention, gate):
    """Independent captured-model reference: materialize its BF16 product."""
    import triton
    import triton.language as tl

    @triton.jit
    def captured_gate(a, g, out, count: tl.constexpr, block: tl.constexpr):
        i = tl.program_id(0) * block + tl.arange(0, block)
        x = tl.load(a + i, i < count, 0).to(tl.float32)
        y = tl.load(g + i, i < count, 0).to(tl.float32)
        tl.store(out + i, x * tl.sigmoid(y), i < count)

    output = attention.new_empty(attention.shape)
    captured_gate[(triton.cdiv(output.numel(), 256),)](
        attention, gate, output, output.numel(), 256
    )
    return output


def _projection_kernel(torch):
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
    )

    kernel = object.__new__(FlashInferFp8DeepGEMMDynamicBlockScaledKernel)
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.fallback = SimpleNamespace(use_deep_gemm_e8m0=True)
    kernel.weight_group_shape = SimpleNamespace(col=128)
    return kernel


def test_cuda_gate_helper_matches_captured_product_rounding(tmp_path):
    """Run the actual combine helper on SM89 too, without pretending to run FA3."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if "CUTLASS_INCLUDE_DIR" not in os.environ or "NVCC" not in os.environ:
        pytest.skip("set NVCC and CUTLASS_INCLUDE_DIR for the isolated helper build")
    include = os.environ["CUTLASS_INCLUDE_DIR"]
    nvcc = os.environ["NVCC"]
    source = tmp_path / "gate_helper.cu"
    library = tmp_path / "gate_helper.so"
    source.write_text(
        """
#include <cuda_runtime.h>
#include "qwen_gate_quant.cuh"
__global__ void gate_helper(cutlass::bfloat16_t const* a,
    cutlass::bfloat16_t const* g, cutlass::float_e4m3_t* q, float* s,
    int rows, int columns) {
    int group = blockIdx.x;
    int i = group * 128 + threadIdx.x * 4;
    float attention[4];
    for (int j = 0; j < 4; ++j) { attention[j] = float(a[i + j]); }
    flash::qwen_gate_quant_warp(attention, g + i, q + i,
        s + group / (columns / 128) + (group % (columns / 128)) * rows);
}
extern "C" int launch_gate_helper(void const* a, void const* g, void* q,
    float* s, int rows, int columns, void* stream) {
    gate_helper<<<rows * columns / 128, 32, 0, (cudaStream_t)stream>>>(
        (cutlass::bfloat16_t const*)a, (cutlass::bfloat16_t const*)g,
        (cutlass::float_e4m3_t*)q, s, rows, columns);
    return int(cudaGetLastError());
}
"""
    )
    major, minor = torch.cuda.get_device_capability()
    subprocess.run(
        [
            nvcc,
            str(source),
            "-o",
            str(library),
            "--shared",
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "-Xcompiler=-fPIC",
            f"-arch=sm_{major}{minor}",
            f"-I{include}",
            f"-I{ROOT / 'tools/flash_attn_fp8'}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    launch = ctypes.CDLL(str(library)).launch_gate_helper
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    torch.manual_seed(29)
    rows, columns = 64, 6144
    attention = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(attention) * 8
    # Distinguish the native CUDA epsilon contract from the Triton pre-clamp.
    attention[0] = 0
    attention[1] *= 1e-11
    quant = torch.empty_like(attention, dtype=torch.float8_e4m3fn)
    scale = torch.empty(columns // 128, rows, device="cuda", dtype=torch.float32).t()
    assert (
        launch(
            attention.data_ptr(),
            gate.data_ptr(),
            quant.data_ptr(),
            scale.data_ptr(),
            rows,
            columns,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    product = _captured_gate_product(attention, gate).float().reshape(rows, -1, 128)
    raw_scale = (product.abs().amax(-1) / 448).clamp_min(1e-10)
    expected_scale = torch.exp2(torch.ceil(torch.log2(raw_scale)))
    expected_q = (
        (product / expected_scale[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(rows, columns)
    )
    assert torch.equal(scale, expected_scale)
    assert torch.equal(quant.view(torch.uint8), expected_q.view(torch.uint8))


@pytest.mark.parametrize("num_splits", [1, 2, 0], ids=["single", "combine", "auto"])
@pytest.mark.parametrize("varlen", [False, True], ids=["q4", "varlen-padded"])
def test_q4_gate_quant_matches_bf16_rounding_and_projection(hopper, num_splits, varlen):
    """Split=2 must exercise real CUDA combine, including dynamic one-split rows."""
    torch, fa = hopper
    from vllm.model_executor.layers.attention.qwen_fp8_output import (
        gate_quant_bf16_attention,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )

    torch.manual_seed(17)
    q_lens = [4] * 8
    if varlen:
        q_lens[-1] = 3
    n = sum(q_lens)
    rows = 32
    q = torch.randn(n, 24, 256, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    # The accepted physical KV layout: [page, head, page_size, 2*head_dim].
    cache = torch.randn(21, 4, 1600, 512, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k, v = cache.transpose(1, 2).split(256, dim=-1)
    gate = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16) * 4
    cu_q = torch.tensor([0, *q_lens], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    seq_k = torch.tensor(
        [17, 1601, 6400, 32768, 32768, 12000, 43, 32000],
        device="cuda",
        dtype=torch.int32,
    )
    block_table = torch.arange(21, device="cuda", dtype=torch.int32).repeat(8, 1)
    descale = torch.full((8, 4), 0.25, device="cuda", dtype=torch.float32)
    kwargs = dict(
        cu_seqlens_q=cu_q,
        seqused_k=seq_k,
        max_seqlen_q=4,
        max_seqlen_k=32768,
        block_table=block_table,
        softmax_scale=256**-0.5,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
        scheduler_metadata=None,
        num_splits=num_splits,
    )
    attention = fa.flash_attn_varlen_func(
        q, k, v, causal=True, fa_version=3, **kwargs
    ).reshape(n, 6144)
    assert attention.dtype == torch.bfloat16
    product = _captured_gate_product(attention, gate[:n].contiguous())
    expected_q, expected_s = per_token_group_quant_fp8(
        product, group_size=128, column_major_scales=True, use_ue8m0=True
    )
    quant = torch.zeros(rows, 6144, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(48, rows, device="cuda", dtype=torch.float32).t()
    fused = fa.flash_attn_varlen_q4_gate_quant(q, k, v, gate, quant, scale, **kwargs)
    assert fused is (num_splits == 2), "do not pass the split-combine test via fallback"
    if not fused:
        gate_quant_bf16_attention(attention, gate, quant, scale, n)
    assert torch.equal(quant[:n].view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale[:n], expected_s)
    assert torch.count_nonzero(quant[n:].view(torch.uint8)) == 0
    assert torch.count_nonzero(scale[n:]) == 0

    weight = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight, weight_scale = deepgemm_post_process_fp8_weight_block(
        wq=weight,
        ws=torch.ones(2, 48, device="cuda", dtype=torch.float32),
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )
    kernel = _projection_kernel(torch)
    padded_product = torch.zeros(rows, 6144, device="cuda", dtype=torch.bfloat16)
    padded_product[:n] = product
    original_projection = kernel.apply_block_scaled_mm(
        padded_product, weight, padded_product.new_empty(1), weight_scale
    )
    projection = kernel.apply_prequantized_mm(quant, scale, weight, weight_scale)
    assert torch.equal(projection[:n], original_projection[:n])


@pytest.mark.parametrize("rows,enabled", [(31, True), (32, False)])
def test_qwen_small_batches_and_default_flag_keep_bf16_projection(
    monkeypatch, rows, enabled
):
    """The model's existing M<32 FlashInfer BF16 input must not be prequantized."""
    import torch

    from vllm.model_executor.models import qwen3_next as model

    monkeypatch.setattr(model.envs, "VLLM_QWEN_FA3_FP8_OUTPUT", enabled)
    monkeypatch.setattr(model.envs, "VLLM_BATCH_INVARIANT", False)
    q = torch.ones(rows, 6144, dtype=torch.bfloat16)
    gate = torch.zeros_like(q)
    kernel = _projection_kernel(torch)

    class QKV(torch.nn.Module):
        quant_method = SimpleNamespace(fp8_linear=None)

        def forward(self, x):
            return x, None

    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return q

    class Projection(torch.nn.Module):
        tp_size = 1
        has_bias = False
        quant_method = SimpleNamespace(fp8_linear=kernel)

        def forward(self, x):
            assert x.dtype == torch.bfloat16
            return x, None

    attention = object.__new__(model.Qwen3NextAttention)
    torch.nn.Module.__init__(attention)
    attention.use_fused_qk_norm_rope_gate = False
    attention.num_heads, attention.head_dim = 24, 256
    attention.qkv_proj, attention.attn, attention.o_proj = (
        QKV(),
        Attention(),
        Projection(),
    )
    monkeypatch.setattr(attention, "_project_qkv_gate", lambda *args: (q, q, q, gate))
    result = attention(torch.arange(rows), q)
    assert torch.equal(result, torch.full_like(q, 0.5))
