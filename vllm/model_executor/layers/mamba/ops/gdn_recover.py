# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Adapted from vLLM's fused_sigmoid_gating.py, which contains MIT-licensed
# flash-linear-attention code (Copyright 2023-2025 Songlin Yang, Yu Zhang).
# Native and mixed producers retain separate CUDA and Triton state kernels.
"""Accepted-state recovery preserving each producer's own arithmetic.

The final state-index column is a private source pointer. A private selector
addresses that column, while preceding columns contain only authorized state
writes. This selector is NOT the sampler acceptance count or convolution bias.
"""

from dataclasses import astuple, dataclass

import torch

from vllm.model_executor.layers.mamba.ops.gdn_recover_native import load_native_recovery
from vllm.triton_utils import tl, triton


@dataclass
class GdnRecoverPayload:
    """Fixed, pre-profile storage in this step's packed spec-token order."""

    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor

    @property
    def nbytes(self) -> int:
        tensors = (self.mixed_qkv, self.a, self.b)
        return sum(t.numel() * t.element_size() for t in tensors)


def allocate_payload(
    *,
    rows: int,
    max_tokens: int,
    num_key_heads: int,
    num_value_heads: int,
    device: torch.device | int,
) -> GdnRecoverPayload:
    """Allocate raw BF16 inputs before the KV pool is sized.

    Args:
        rows: Maximum real request count, not the prefill-token budget.
        max_tokens: Maximum verifier window including the compulsory token.
        num_key_heads: TP-local key heads.
        num_value_heads: TP-local value heads.
        device: Worker CUDA device.

    """
    # Register/load the native extension before profiling or CUDA graph capture.
    load_native_recovery()
    capacity = rows * max_tokens
    width = (2 * num_key_heads + num_value_heads) * 128
    return GdnRecoverPayload(
        torch.empty((capacity, width), dtype=torch.bfloat16, device=device),
        torch.empty((capacity, num_value_heads), dtype=torch.bfloat16, device=device),
        torch.empty((capacity, num_value_heads), dtype=torch.bfloat16, device=device),
    )


def recover_native(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    source_selectors: torch.Tensor,
    state_pool: torch.Tensor,
    out: torch.Tensor | None = None,
    num_key_heads: int,
    num_value_heads: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    output_gate_activation: str = "silu",
    scale: float | None = None,
) -> None:
    """Replay accepted native states with the producer's CUDA recurrence.

    Output-only arguments remain accepted for compatibility with existing callers.
    Compile once before graph capture; no output scratch is read or written.
    """
    tokens, rows = mixed_qkv.shape[0], state_indices.shape[0]
    if tokens == 0 or rows == 0:
        return
    if (num_key_heads, num_value_heads) != (16, 48):
        raise ValueError("Native state-only candidate requires TP1 Hk=16/Hv=48")
    load_native_recovery()
    torch.ops.gdn_recover_state_only.native(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        source_selectors,
        state_pool,
    )


def recover_triton(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    source_selectors: torch.Tensor,
    state_pool: torch.Tensor,
    out: torch.Tensor | None = None,
    num_key_heads: int,
    num_value_heads: int,
    scale: float | None = None,
) -> None:
    """Replay only required mixed-route states, without QKV copies or output.

    ``out`` and ``scale`` remain accepted for compatibility with existing callers, but
    neither enters the state recurrence. Native CUDA replay stays separate.
    """
    tokens, rows = mixed_qkv.shape[0], state_indices.shape[0]
    if tokens == 0 or rows == 0:
        return
    recover_triton_state_kernel[(1, 4, rows * num_value_heads)](
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
        mixed_qkv=mixed_qkv,
        state=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        source_selectors=source_selectors,
        H=num_key_heads,
        HV=num_value_heads,
        K=128,
        V=128,
        BK=128,
        BV=32,
        INDEX_BLOCK=triton.next_power_of_2(state_indices.shape[1]),
        index_width=state_indices.shape[1],
        mixed_row_stride=mixed_qkv.stride(0),
        a_row_stride=a.stride(0),
        a_head_stride=a.stride(1),
        b_row_stride=b.stride(0),
        b_head_stride=b.stride(1),
        state_slot_stride=state_pool.stride(0),
        indices_row_stride=state_indices.stride(0),
        indices_token_stride=state_indices.stride(1),
        num_warps=4,
        num_stages=3,
    )


@triton.jit
def recover_triton_state_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    mixed_qkv,
    state,
    cu_seqlens,
    state_indices,
    source_selectors,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    INDEX_BLOCK: tl.constexpr,
    index_width: tl.constexpr,
    mixed_row_stride: tl.constexpr,
    a_row_stride: tl.constexpr,
    a_head_stride: tl.constexpr,
    b_row_stride: tl.constexpr,
    b_head_stride: tl.constexpr,
    state_slot_stride: tl.constexpr,
    indices_row_stride: tl.constexpr,
    indices_token_stride: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    length = eos - bos
    if length == 0:
        return

    source_column = tl.load(source_selectors + i_n).to(tl.int64) - 1
    source_slot = tl.load(
        state_indices + i_n * indices_row_stride + source_column * indices_token_stride
    ).to(tl.int64)
    if source_slot <= 0:
        return

    # Writes are sparse: a block-boundary snapshot and the final accepted state.
    # The source selector column is not a write, even for a short/padded row.
    columns = tl.arange(0, INDEX_BLOCK)
    destinations = tl.load(
        state_indices + i_n * indices_row_stride + columns * indices_token_stride,
        mask=(columns < index_width) & (columns < length) & (columns != source_column),
        other=0,
    )
    replay_length = tl.max(tl.where(destinations > 0, columns + 1, 0), 0)
    if replay_length == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_k = mixed_qkv + bos * mixed_row_stride + (H + i_h) * K + o_k
    p_v = mixed_qkv + bos * mixed_row_stride + 2 * H * K + i_hv * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * a_row_stride + i_hv * a_head_stride
    p_dt_bias = dt_bias + i_hv
    p_b = b + bos * b_row_stride + i_hv * b_head_stride
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    state_offsets = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    p_h0 = state + source_slot * state_slot_stride + state_offsets
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(replay_length):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
        b_beta = tl.sigmoid(b_b.to(tl.float32))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]

        final_state_idx = tl.load(
            state_indices + i_n * indices_row_stride + i_t * indices_token_stride
        ).to(tl.int64)
        if final_state_idx > 0:
            p_ht = state + final_state_idx * state_slot_stride + state_offsets
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_k += mixed_row_stride
        p_v += mixed_row_stride
        p_b += b_row_stride
        p_a += a_row_stride


@dataclass(frozen=True)
class GdnRecoverLayerDescriptor:
    # This int64 layout matches GdnRecoverLayerDescriptor in the CUDA extension.
    mixed_qkv: int
    a: int
    b: int
    a_log: int
    dt_bias: int
    state: int
    conv: int
    mixed_row: int
    a_row: int
    b_row: int
    state_slot: int
    dt_bias_type: int
    conv_page: int
    conv_channel: int
    conv_time: int
    history: int


class GdnRecoverBatchStorage:
    """One 6144-byte GPU allocation charged during model loading, before profiling.

    Group membership is supplied by the owning metadata builders, not inferred
    from model order. The actual TP1 model has three groups of sixteen GDN layers.
    """

    def __init__(self, device: torch.device | int) -> None:
        load_native_recovery()
        self.descriptors = torch.empty((3, 16, 16), dtype=torch.int64, device=device)
        self.host_descriptors = torch.empty(
            (3, 16, 16), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.group_slots: dict[frozenset[str], int] = {}

    def group(self, layer_names: list[str]) -> "GdnRecoverBatch":
        key = frozenset(layer_names)
        num_layers = len(layer_names)
        assert num_layers == len(key) and 0 < num_layers <= 16
        if key not in self.group_slots:
            assert len(self.group_slots) < 3, "Expected three GDN cache groups"
            self.group_slots[key] = len(self.group_slots)
        slot = self.group_slots[key]
        return GdnRecoverBatch(
            self.descriptors[slot, :num_layers],
            self.host_descriptors[slot, :num_layers],
        )


class GdnRecoverBatch:
    """One descriptor view per attention builder, not per step.

    The builder exists before minimal/real KV binding and installs the layer
    binding callbacks. No pool tensor is retained here: teardown can free the
    minimal pool, and every subsequent bind replaces that layer's complete row.
    """

    def __init__(
        self, descriptors: torch.Tensor, host_descriptors: torch.Tensor
    ) -> None:
        self.descriptors = descriptors
        self._host_descriptors = host_descriptors
        self._bound_layers: set[int] = set()

    @property
    def nbytes(self) -> int:
        return self.descriptors.numel() * self.descriptors.element_size()

    def bind_layer(
        self,
        index: int,
        *,
        payload: GdnRecoverPayload,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        conv: torch.Tensor,
        history: int,
    ) -> None:
        """Called by layer.bind_kv_cache, never from metadata.build or commit."""
        assert state.dtype == torch.float32 and state.shape[1:] == (48, 128, 128)
        assert state.stride()[1:] == (128 * 128, 128, 1)
        assert state.stride(0) % 4 == 0 and state.data_ptr() % 16 == 0
        assert conv.dtype == torch.bfloat16 and conv.shape[1] == 10240
        assert history == 3 and conv.shape[2] >= history + 3
        assert A_log.dtype == torch.float32 and A_log.is_contiguous()
        assert A_log.numel() == 48 and dt_bias.is_contiguous()
        assert dt_bias.numel() == 48
        assert payload.mixed_qkv.shape[1] == 10240
        assert payload.a.shape[1] == payload.b.shape[1] == 48
        assert all(
            t.dtype == torch.bfloat16 and t.stride(1) == 1
            for t in (payload.mixed_qkv, payload.a, payload.b)
        )
        tensors = (payload.mixed_qkv, payload.a, payload.b, A_log, dt_bias, state, conv)
        assert all(t.device == self.descriptors.device for t in tensors)
        bias_type = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[
            dt_bias.dtype
        ]
        descriptor = GdnRecoverLayerDescriptor(
            payload.mixed_qkv.data_ptr(),
            payload.a.data_ptr(),
            payload.b.data_ptr(),
            A_log.data_ptr(),
            dt_bias.data_ptr(),
            state.data_ptr(),
            conv.data_ptr(),
            payload.mixed_qkv.stride(0),
            payload.a.stride(0),
            payload.b.stride(0),
            state.stride(0),
            bias_type,
            conv.stride(0),
            conv.stride(1),
            conv.stride(2),
            history,
        )
        # Initialization-only CPU construction and H2D; hot steps only read it.
        self._host_descriptors[index].copy_(
            torch.tensor(astuple(descriptor), dtype=torch.int64, device="cpu")
        )
        self.descriptors[index].copy_(self._host_descriptors[index], non_blocking=True)
        self._bound_layers.add(index)

    def recover_native(
        self,
        state_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        source_selectors: torch.Tensor,
    ) -> None:
        assert len(self._bound_layers) == self.descriptors.shape[0]
        torch.ops.gdn_recover_state_only.native_batch(
            self.descriptors, state_indices, cu_seqlens, source_selectors
        )

    def compact_conv(self, plan: torch.Tensor) -> None:
        assert len(self._bound_layers) == self.descriptors.shape[0]
        torch.ops.gdn_recover_state_only.conv_batch(self.descriptors, plan)
