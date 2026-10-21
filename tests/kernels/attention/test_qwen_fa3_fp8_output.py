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
    attention[2] = 0
    gate[2] = 64
    attention[2, 0], gate[2, 0] = 187 / 128, 0.5
    attention[2, 1] = 128
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


def test_bf16_fallback_quantizes_flat_queries_before_head_view(monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention import qwen_fp8_output
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    rows, columns, actual = 64, 6144, 63
    generator = torch.Generator(device="cuda").manual_seed(41)
    attention = torch.randn(
        rows, columns, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    gate = (
        torch.randn(
            rows, columns, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 8
    )
    attention[0] = 0
    attention[1] *= 1e-11
    attention[2] = 7
    gate[2] = 64
    attention[3] = 0
    gate[3] = 64
    attention[3, 0], gate[3, 0] = 187 / 128, 0.5
    attention[3, 1] = 128
    product = _captured_gate_product(attention[:actual], gate[:actual])
    expected_q = torch.empty_like(product, dtype=torch.float8_e4m3fn)
    expected_scale = torch.empty(
        columns // 128, actual, dtype=torch.float32, device="cuda"
    ).t()
    torch.ops._C.per_token_group_fp8_quant(
        product, expected_q, expected_scale, 128, 1e-10, -448, 448, True, True, False
    )
    quant = torch.empty_like(attention, dtype=torch.float8_e4m3fn)
    scale = torch.empty(columns // 128, rows, dtype=torch.float32, device="cuda").t()
    with set_current_vllm_config(VllmConfig()):
        query_quant = QuantFP8(
            static=True, group_shape=GroupShape.PER_TENSOR, compile_native=False
        )

    def backend_forward(layer, query, key, value, kv_cache, metadata, *, output):
        assert query.dtype == torch.float8_e4m3fn
        assert query.shape == (rows, 24, 256)
        output.copy_(attention.view_as(output))

    layer = SimpleNamespace(
        num_heads=24,
        head_size=256,
        num_kv_heads=4,
        head_size_v=256,
        query_quant=query_quant.forward_cuda,
        _q_scale=torch.ones(1, device="cuda"),
        impl=SimpleNamespace(supports_quant_query_input=True, forward=backend_forward),
        attn_backend=SimpleNamespace(forward_includes_kv_cache_update=True),
    )
    metadata = SimpleNamespace(num_actual_tokens=actual)
    monkeypatch.setattr(
        qwen_fp8_output,
        "get_attention_context",
        lambda _: (metadata, layer, None, None),
    )
    key = torch.zeros(rows, 1024, dtype=torch.bfloat16, device="cuda")
    torch.ops.vllm.qwen_fa3_gate_quant(
        attention, key, key, gate, quant, scale, "attention"
    )
    assert torch.equal(quant[:actual].view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale[:actual], expected_scale)
    assert quant[3, 0].float().item() == 1.875
    assert quant[actual:].view(torch.uint8).count_nonzero().item() == 0
    assert scale[actual:].count_nonzero().item() == 0


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
    # Default LBNHC storage: [page, token, head, 2*head_dim].
    cache = torch.randn(21, 1600, 4, 512, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k, v = cache.split(256, dim=-1)
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
    provided = torch.empty_like(projection)
    returned = kernel.apply_prequantized_mm(
        quant, scale, weight, weight_scale, output=provided
    )
    assert returned is provided
    assert torch.equal(provided[:n], original_projection[:n])


@pytest.mark.parametrize(
    "rows,enabled,raw_input",
    [
        (31, True, False),
        (32, False, False),
        (32, True, False),
        (32, True, True),
        (193, True, False),
        (16384, True, False),
    ],
)
def test_qwen_preserves_bf16_projection_routes(monkeypatch, rows, enabled, raw_input):
    """Keep eager, non-decode and raw-activation projection routes unchanged."""
    import torch

    from vllm.model_executor.models import qwen3_next as model

    monkeypatch.setattr(model.envs, "VLLM_QWEN_FA3_FP8_OUTPUT", enabled)
    monkeypatch.setattr(model.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: raw_input or rows > 192)
    monkeypatch.setattr(
        model,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode=model.CompilationMode.VLLM_COMPILE, backend="inductor"
            )
        ),
    )
    monkeypatch.setattr(
        model,
        "qwen_attention_fp8_output",
        lambda *args: pytest.fail("eager attention must not use the compiled gate ABI"),
    )
    q = torch.ones(rows, 6144, dtype=torch.bfloat16)
    gate = torch.zeros_like(q)
    q[:, 0], gate[:, 0] = 187 / 128, 0.5
    kernel = _projection_kernel(torch)

    class QKV(torch.nn.Module):
        quant_method = SimpleNamespace(fp8_linear=None)

        def forward(self, x):
            return x, None

    class Attention(torch.nn.Module):
        layer_name = "attention"

        def forward(self, q, k, v):
            return q

    class Projection(torch.nn.Module):
        tp_size = 1
        has_bias = False
        requires_unquantized_input = raw_input
        if not raw_input:
            quant_method = SimpleNamespace(fp8_linear=kernel)

        def forward(self, x):
            assert x.dtype == torch.bfloat16
            return x + 1 if raw_input else x, None

    attention = object.__new__(model.Qwen3NextAttention)
    torch.nn.Module.__init__(attention)
    attention.use_fused_qk_norm_rope_gate = False
    attention._fp8_output_device_supported = True
    attention._max_q4_decode_tokens = 192
    attention.num_heads, attention.head_dim = 24, 256
    attention.qkv_proj, attention.attn, attention.o_proj = (
        QKV(),
        Attention(),
        Projection(),
    )
    monkeypatch.setattr(attention, "_project_qkv_gate", lambda *args: (q, q, q, gate))
    result = attention(torch.arange(rows), q)
    expected = torch.full_like(q, 0.5)
    expected[:, 0] = 232 / 256
    if raw_input:
        expected += 1
    assert torch.equal(result, expected)
