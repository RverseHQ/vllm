# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Qwen attention gate/FP8 output for the existing DeepGEMM input."""

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType


@triton.jit
def _gate_quant_bf16_attention(
    attention,
    gate,
    quant,
    scale,
    actual_rows,
    gate_row_stride: tl.constexpr,
    rows: tl.constexpr,
    columns: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    col = group * 128 + tl.arange(0, 128)
    a = tl.load(attention + row * columns + col, row < actual_rows, 0).to(tl.float32)
    g = tl.load(gate + row * gate_row_stride + col, row < actual_rows, 0).to(tl.float32)
    # The compiled model keeps sigmoid in FP32 and rounds only the product.
    product = (a * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    raw_scale = tl.maximum(tl.max(tl.abs(product), 0) / 448.0, 1e-10)
    bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    s = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(product / s, -448.0), 448.0)
    tl.store(quant + row * columns + col, q)
    tl.store(scale + row + group * rows, tl.where(row < actual_rows, s, 0.0))


def gate_quant_bf16_attention(
    attention: torch.Tensor,
    gate: torch.Tensor,
    quant: torch.Tensor,
    scale: torch.Tensor,
    actual_rows: int,
) -> None:
    """Fallback matching the captured BF16 attention / FP32 sigmoid contract."""
    rows, columns = quant.shape
    _gate_quant_bf16_attention[(rows, columns // 128)](
        attention,
        gate,
        quant,
        scale,
        actual_rows,
        gate.stride(0),
        rows,
        columns,
        num_warps=4,
    )


def try_q4_gate_quant(
    layer,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata,
    gate: torch.Tensor,
    quant: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Use only the existing FA3 Q4 cohort; other attention contracts stay BF16."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_q4_gate_quant,
    )

    impl = layer.impl
    if not (
        isinstance(impl, FlashAttentionImpl)
        and metadata is not None
        and impl.vllm_flash_attn_version == 3
        and impl.attn_type == AttentionType.DECODER
        and impl.dcp_world_size == 1
        and metadata.causal is True
        and impl.sliding_window == (-1, -1)
        and impl.logits_soft_cap == 0
        and impl.alibi_slopes is None
        and impl.sinks is None
        and metadata.mm_prefix_query_range_tensor is None
        and metadata.rswa_prefix_lens is None
        and metadata.max_num_splits > 1
        and query.dtype == torch.float8_e4m3fn
        and kv_cache.element_size() == 1
    ):
        return False
    key_cache, value_cache = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
    key_cache = key_cache.view(torch.float8_e4m3fn)
    value_cache = value_cache.view(torch.float8_e4m3fn)
    n = metadata.num_actual_tokens
    descale_shape = (metadata.query_start_loc.shape[0] - 1, impl.num_kv_heads)
    fused = flash_attn_varlen_q4_gate_quant(
        query[:n],
        key_cache,
        value_cache,
        gate,
        quant,
        scale,
        cu_seqlens_q=metadata.query_start_loc,
        seqused_k=metadata.seq_lens,
        max_seqlen_q=metadata.max_query_len,
        max_seqlen_k=metadata.max_seq_len,
        block_table=metadata.block_table,
        softmax_scale=impl.scale,
        q_descale=layer._q_scale.expand(descale_shape),
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        scheduler_metadata=metadata.scheduler_metadata,
        num_splits=metadata.max_num_splits,
    )
    if fused and n < quant.shape[0]:
        quant[n:].zero_()
        scale[n:].zero_()
    return fused


@eager_break_during_capture
@maybe_transfer_kv_layer
def qwen_fa3_gate_quant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    quant: torch.Tensor,
    scale: torch.Tensor,
    layer_name: str,
) -> None:
    metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    query = query.view(-1, layer.num_heads, layer.head_size)
    key = key.view(-1, layer.num_kv_heads, layer.head_size)
    value = value.view(-1, layer.num_kv_heads, layer.head_size_v)
    if layer.query_quant is not None and layer.impl.supports_quant_query_input:
        query, _ = layer.query_quant(query, layer._q_scale)
    if (
        not layer.attn_backend.forward_includes_kv_cache_update
        and layer.kv_sharing_target_layer_name is None
    ):
        unified_kv_cache_update(key, value, layer_name)
    if try_q4_gate_quant(layer, query, kv_cache, metadata, gate, quant, scale):
        return
    attention = torch.empty(
        (query.shape[0], layer.num_heads, layer.head_size_v),
        dtype=torch.bfloat16,
        device=query.device,
    )
    layer.impl.forward(layer, query, key, value, kv_cache, metadata, output=attention)
    actual_rows = metadata.num_actual_tokens if metadata is not None else query.shape[0]
    gate_quant_bf16_attention(attention, gate, quant, scale, actual_rows)


direct_register_custom_op(
    op_name="qwen_fa3_gate_quant",
    op_func=qwen_fa3_gate_quant,
    mutates_args=["quant", "scale"],
)


def qwen_attention_fp8_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = gate.shape
    quant = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=query.device)
    scale = torch.empty(
        (columns // 128, rows), dtype=torch.float32, device=query.device
    ).t()
    torch.ops.vllm.qwen_fa3_gate_quant(
        query, key, value, gate, quant, scale, layer_name
    )
    return quant, scale
