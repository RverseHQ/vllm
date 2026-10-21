# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-path tests for fused Qwen3.5 GDN MTP decode."""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.has_device_capability(80)):
    pytest.skip(
        reason="Fused GDN MTP decode requires CUDA compute capability 8.0+.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
SPEC_TOKENS = NUM_SPEC + 1
H = 1
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
EPS = 1e-6


class _TestGatedNorm:
    def __init__(self, weight: torch.Tensor, activation: str) -> None:
        self.weight = weight
        self.bias = None
        self.eps = EPS
        self.group_size = None
        self.norm_before_gate = True
        self.activation = activation

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x,
            self.weight,
            None,
            z=z,
            eps=self.eps,
            norm_before_gate=True,
            activation=self.activation,
        )


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.additional_config = {"gdn_prefill_backend": "cutedsl"}
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_layer(
    vllm_config,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    output_gate_activation: str,
):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=K,
        value_dim=HV * V,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_TestGatedNorm(norm_weight, output_gate_activation),
        layer_norm_epsilon=EPS,
        gdn_decode_kernel="cuda",
        use_gdn_recovery=False,
        num_spec=NUM_SPEC,
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "_forward_core_fused_norm_fp8_packed",
        "_gdn_fp8_output_kernel",
        "forward_cuda",
        "split_ba",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@pytest.mark.parametrize(
    "num_v_heads,expected",
    [
        pytest.param(2, True, id="ratio1"),
        pytest.param(4, True, id="ratio2"),
        pytest.param(6, True, id="ratio3"),
        pytest.param(8, True, id="ratio4"),
        pytest.param(16, True, id="ratio8"),
        pytest.param(5, False, id="non-integral-ratio"),
        pytest.param(10, False, id="unsupported-ratio5"),
    ],
)
def test_fused_mtp_head_ratio_guard(num_v_heads: int, expected: bool) -> None:
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    layer = types.SimpleNamespace(
        num_k_heads=2,
        num_v_heads=num_v_heads,
        kv_cache=(None, torch.empty(1, dtype=torch.float32, device="cuda")),
        gdn_decode_kernel="cuda",
        use_gdn_recovery=False,
    )
    attn_metadata = types.SimpleNamespace(
        spec_state_indices_tensor=torch.ones(
            1, SPEC_TOKENS, dtype=torch.int32, device="cuda"
        ),
        spec_sequence_masks=object(),
        num_decodes=0,
        num_spec_decodes=1,
    )

    assert (
        QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode(
            cast(QwenGatedDeltaNetAttention, layer),
            cast(GDNAttentionMetadata, attn_metadata),
        )
        is expected
    )


@torch.inference_mode()
def test_fused_forward_uses_packed_entrypoint() -> None:
    """Fused mode keeps projected QKVZ and BA packed through the model op."""
    device = torch.device("cuda")
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=device)
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_fused_gdn_spec_decode=True,
        norm=types.SimpleNamespace(
            weight=torch.empty(V, dtype=torch.bfloat16, device=device)
        ),
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        in_proj_qkvz=lambda _: (mixed_qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        out_proj=lambda x: (x, None),
    )
    layer.forward_cuda = types.MethodType(
        QwenGatedDeltaNetAttention.forward_cuda, layer
    )

    def packed_op(
        actual_qkvz: torch.Tensor,
        actual_ba: torch.Tensor,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        assert actual_qkvz is mixed_qkvz
        assert actual_ba is ba
        assert layer_name == _encode_layer_name(PREFIX)
        output.fill_(1)

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed_mock,
        patch.object(torch.ops.vllm, "qwen_gdn_attention_core") as triton_mock,
    ):
        output = layer.forward_cuda(hidden_states)

    packed_mock.assert_called_once()
    triton_mock.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(output))


@pytest.mark.parametrize(
    "seq_lens,query_lens,draft_tokens,expected_fused_calls",
    [
        pytest.param([128], [SPEC_TOKENS], [NUM_SPEC], 1, id="pure-mtp"),
        pytest.param(
            [128, 96],
            [SPEC_TOKENS, 64],
            [NUM_SPEC, -1],
            0,
            id="mixed-mtp-falls-back",
        ),
        pytest.param([96], [64], [-1], 0, id="pure-prefill"),
        pytest.param([128], [1], [-1], 0, id="pure-decode"),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_model_path_matches_reference(
    seq_lens: list[int],
    query_lens: list[int],
    draft_tokens: list[int],
    expected_fused_calls: int,
    output_gate_activation: str,
) -> None:
    """Fused MTP and its mixed/prefill/decode fallbacks match the reference."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(draft_tokens, dtype=torch.int32),
        )

    state_indices = [
        indices
        for indices in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if indices is not None and indices.numel() > 0
    ]
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = 0.01 * torch.randn(
        pool_size, *temporal_state_shape, dtype=torch.float32, device=device
    )
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkv = 0.1 * torch.randn(
        num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)
    output_gate = 0.1 * torch.randn(
        num_tokens, HV, V, dtype=torch.bfloat16, device=device
    )
    mixed_qkvz = torch.cat((mixed_qkv, output_gate.flatten(1)), dim=-1)
    ba = torch.cat((b, a), dim=-1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    reference_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    reference_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        reference_layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b,
            a=a,
            core_attn_out=reference_out,
        )
    reference_out = reference_layer.norm(reference_out, output_gate)

    fused_layer = _build_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        output_gate_activation,
    )
    context.no_compile_layers = {PREFIX: fused_layer}
    fused_out = torch.zeros_like(output_gate)
    fused_op = qwen_gdn_linear_attn.ops.fused_gdn_decode_post_conv_mtp
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
        patch.object(
            qwen_gdn_linear_attn.ops,
            "fused_gdn_decode_post_conv_mtp",
            wraps=fused_op,
        ) as fused_mock,
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            fused_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    assert fused_mock.call_count == expected_fused_calls
    torch.testing.assert_close(
        fused_layer.kv_cache[0], reference_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(fused_out, reference_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        reference_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize("rows,padding", [(32, 0), (32, 4), (192, 0), (192, 4)])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_native_fp8_output_matches_official_quantizer(
    rows: int, padding: int, activation: str, norm_dtype: torch.dtype
) -> None:
    """Fusion preserves BF16 rounding, column scales and every FP32 state byte."""
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 conversion probe requires compute capability 8.9+")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp_fp8"):
        pytest.skip("fused GDN FP8 output op is not built")
    from benchmarks.kernels.benchmark_gdn_fp8_output import check_native_output

    check_native_output(rows, padding, activation, norm_dtype, installed=True)


@pytest.mark.parametrize(
    "rows,expected", [(4, False), (31, False), (32, True), (192, True)]
)
def test_gdn_fp8_output_preserves_small_batch_route(rows: int, expected: bool) -> None:
    """Only the existing group128/E8M0 DeepGEMM branch can consume fused output."""
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp_fp8"):
        pytest.skip("fused GDN FP8 output op is not built")
    kernel_type = qwen_gdn_linear_attn.FlashInferFp8DeepGEMMDynamicBlockScaledKernel
    kernel = object.__new__(kernel_type)
    kernel.weight_group_shape = types.SimpleNamespace(col=128)
    kernel.fallback = types.SimpleNamespace(use_deep_gemm_e8m0=True)
    method = object.__new__(qwen_gdn_linear_attn.Fp8LinearMethod)
    method.fp8_linear = kernel
    layer = types.SimpleNamespace(
        tp_size=1, out_proj=types.SimpleNamespace(quant_method=method, bias=None)
    )
    with (
        patch.object(current_platform, "has_device_capability", return_value=True),
        patch.object(qwen_gdn_linear_attn.envs, "VLLM_BATCH_INVARIANT", False),
    ):
        actual = QwenGatedDeltaNetAttention._gdn_fp8_output_kernel(layer, rows)
    assert (actual is kernel) == expected


def test_gdn_fp8_output_preserves_raw_activation_consumers() -> None:
    layer = types.SimpleNamespace(
        tp_size=1,
        out_proj=types.SimpleNamespace(requires_unquantized_input=True),
    )
    assert QwenGatedDeltaNetAttention._gdn_fp8_output_kernel(layer, 192) is None


@pytest.mark.parametrize(
    "rows,padding,route",
    [
        (32, 0, "native"),
        (192, 4, "native"),
        (64, 0, "mixed"),
        (64, 0, "prefill"),
        (32, 0, "bf16-state"),
    ],
)
@torch.inference_mode()
def test_fp8_model_path_matches_native_projection(
    rows: int, padding: int, route: str
) -> None:
    """The enabled model consumer matches native BF16 + official DeepGEMM."""
    if not current_platform.has_device_capability(90):
        pytest.skip("The production DeepGEMM projection requires SM90+")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp_fp8"):
        pytest.skip("fused GDN FP8 output op is not built")
    from benchmarks.kernels.benchmark_gdn_fp8_output import assert_bytes_equal
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    torch.manual_seed(73)
    device = torch.device("cuda")
    actual = rows - padding
    requests = actual // SPEC_TOKENS
    config = _make_vllm_config()
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=config,
        device=device,
    )
    if route == "mixed":
        query_lens = [SPEC_TOKENS] * (requests // 2) + [actual // 2]
        drafts = [NUM_SPEC] * (requests // 2) + [-1]
    elif route == "prefill":
        query_lens, drafts = [actual], [-1]
    else:
        query_lens, drafts = [SPEC_TOKENS] * requests, [NUM_SPEC] * requests
    batch = BatchSpec(seq_lens=[128] * len(query_lens), query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                len(query_lens), dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        )
    indices = [
        tensor
        for tensor in (
            metadata.spec_state_indices_tensor,
            metadata.non_spec_state_indices_tensor,
        )
        if tensor is not None and tensor.numel()
    ]
    slots = max(int(tensor.max().item()) for tensor in indices) + 1
    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
    )
    conv = torch.randn(slots, *conv_shape, dtype=torch.bfloat16, device=device) * 0.05
    state_dtype = torch.bfloat16 if route == "bf16-state" else torch.float32
    state = torch.randn(slots, *state_shape, dtype=state_dtype, device=device) * 0.01
    a_log, bias = torch.randn(2, HV, device=device) * 0.1
    conv_weight = (
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device) * 0.1
    )
    norm = torch.randn(V, device=device)
    qkvz = (
        torch.randn(rows, CONV_DIM + HV * V, dtype=torch.bfloat16, device=device) * 0.1
    )
    ba = torch.randn(rows, 2 * HV, dtype=torch.bfloat16, device=device) * 0.1
    hidden = torch.empty(rows, 1, dtype=torch.bfloat16, device=device)

    kernel_type = qwen_gdn_linear_attn.FlashInferFp8DeepGEMMDynamicBlockScaledKernel
    kernel = object.__new__(kernel_type)
    kernel.config = types.SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.weight_group_shape = types.SimpleNamespace(col=128)
    kernel.fallback = types.SimpleNamespace(use_deep_gemm_e8m0=True)
    method = object.__new__(qwen_gdn_linear_attn.Fp8LinearMethod)
    method.fp8_linear = kernel
    weight, weight_scale = deepgemm_post_process_fp8_weight_block(
        wq=(torch.randn(128, HV * V, device=device) * 0.1).to(torch.float8_e4m3fn),
        ws=torch.ones(1, HV, device=device),
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )

    class Projection:
        bias = None
        output_size = 128
        quant_method = method
        weight_scale_inv = weight_scale

        def __init__(self):
            self.weight = weight

        def __call__(self, x):
            return kernel.apply_weights(self, x), None

    outputs, states, conv_states = [], [], []
    for enabled in (False, True):
        layer = _build_layer(
            config, conv.clone(), state.clone(), a_log, bias, conv_weight, norm, "silu"
        )
        layer.enable_fused_gdn_spec_decode = True
        layer.in_proj_qkvz = lambda _: (qkvz.clone(), None)
        layer.in_proj_ba = lambda _: (ba, None)
        layer.out_proj = Projection()
        context = types.SimpleNamespace(
            attn_metadata={PREFIX: metadata}, no_compile_layers={PREFIX: layer}
        )
        with (
            patch.object(
                qwen_gdn_linear_attn, "get_forward_context", return_value=context
            ),
            patch.object(
                qwen_gdn_linear_attn.envs, "VLLM_QWEN_GDN_FP8_OUTPUT", enabled
            ),
            patch.object(
                qwen_gdn_linear_attn.ops,
                "fused_gdn_decode_post_conv_mtp_fp8",
                wraps=qwen_gdn_linear_attn.ops.fused_gdn_decode_post_conv_mtp_fp8,
            ) as fused,
        ):
            outputs.append(layer.forward_cuda(hidden))
        assert fused.call_count == int(enabled and route == "native")
        states.append(layer.kv_cache[1])
        conv_states.append(layer.kv_cache[0])
    assert_bytes_equal(outputs[1], outputs[0])
    assert_bytes_equal(states[1], states[0])
    assert_bytes_equal(conv_states[1], conv_states[0])
