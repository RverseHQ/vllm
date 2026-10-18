# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact speculative GDN replay with one physical running-state slot."""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.mamba.ops.gdn_recover import GdnRecoverBatch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.kv_cache_interface import MambaSpec


class GDNRecoverAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVER_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNRecoverAttentionMetadataBuilder"]:
        return GDNRecoverAttentionMetadataBuilder


@dataclass(kw_only=True)
class GDNRecoverAttentionMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    layers: tuple["QwenGatedDeltaNetAttention", ...]
    recovery_batch: GdnRecoverBatch
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    request_indices: torch.Tensor
    verify_indices: torch.Tensor
    source_selectors: torch.Tensor
    replay_indices: torch.Tensor
    plan: torch.Tensor
    final_columns: torch.Tensor
    block_size: int
    max_tokens: int

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        if self.num_spec_decodes == 0:
            return None
        assert self.spec_state_indices_tensor is not None
        assert self.spec_query_start_loc is not None
        n = self.num_spec_decodes
        plan_gdn_recovery(
            num_accepted_tokens=num_accepted_tokens,
            request_indices=self.request_indices,
            num_computed_tokens=self.num_computed_tokens,
            block_table=self.block_table,
            source_slots=self.spec_state_indices_tensor[:n, 0],
            cu_seqlens=self.spec_query_start_loc[: n + 1],
            replay_indices=self.replay_indices[:n],
            plan=self.plan[:n],
            final_columns=self.final_columns[:n],
            block_size=self.block_size,
            max_tokens=self.max_tokens,
        )
        if all(layer.uses_native_gdn_recovery(self) for layer in self.layers):
            self.recovery_batch.recover_native(
                self.replay_indices[:n],
                self.spec_query_start_loc[: n + 1],
                self.source_selectors[:n],
            )
        else:
            # Mixed prefill retains the producer's separate Triton arithmetic.
            for layer in self.layers:
                layer.commit_gdn_recovery(self, compact_conv=False)
        self.recovery_batch.compact_conv(self.plan[:n])
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=n,
            request_indices=self.request_indices,
            block_table=self.block_table,
            num_computed_tokens=self.num_computed_tokens,
            block_size=self.block_size,
            running_state_columns=self.final_columns[:n],
        )


class GDNRecoverAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: "MambaSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert kv_cache_spec.num_speculative_blocks == 0
        context = vllm_config.compilation_config.static_forward_context
        self.layers: tuple[QwenGatedDeltaNetAttention, ...] = tuple(
            context[name] for name in layer_names
        )
        assert self.num_spec == 3, "Batched GDN recovery currently requires MTP3"
        # init_attn_backend creates one builder per cache group (3 x 16 here).
        # Both profiling and real initialization create a new builder before
        # bind_kv_cache; layer callbacks fill this group's descriptors then.
        self.recovery_batch = self.layers[0].gdn_recovery_storage.group(layer_names)
        for index, layer in enumerate(self.layers):
            layer.gdn_recovery_binding = (weakref.ref(self.recovery_batch), index)
        n = self.decode_cudagraph_max_bs
        self.verify_indices = torch.zeros(
            (n, self.num_spec + 2), dtype=torch.int32, device=device
        )
        # This is the old kernel's source-column selector, NOT sampler acceptance.
        self.source_selectors = torch.full(
            (n,), self.num_spec + 2, dtype=torch.int32, device=device
        )
        self.replay_indices = torch.empty_like(self.verify_indices)
        self.plan = torch.empty((n, 5), dtype=torch.int32, device=device)
        self.final_columns = torch.empty((n,), dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ) -> GDNRecoverAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
        )
        n = metadata.num_spec_decodes
        request_indices_cpu = torch.empty(0, dtype=torch.int32)
        if n:
            assert num_decode_draft_tokens_cpu is not None
            request_indices_cpu = torch.nonzero(
                num_decode_draft_tokens_cpu >= 0, as_tuple=True
            )[0].to(torch.int32)
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.num_accepted_tokens is not None
            self.verify_indices[:n, -1].copy_(metadata.spec_state_indices_tensor[:n, 0])
            self.verify_indices[n:, -1].zero_()
            # The prior commit compacted convolution history to offset zero.
            metadata.num_accepted_tokens.fill_(1)
        return GDNRecoverAttentionMetadata(
            **vars(metadata),
            layers=self.layers,
            recovery_batch=self.recovery_batch,
            block_table=common_attn_metadata.block_table_tensor,
            num_computed_tokens=common_attn_metadata.compute_num_computed_tokens(),
            request_indices=async_tensor_h2d(
                request_indices_cpu, device=self.verify_indices.device
            ),
            verify_indices=self.verify_indices,
            source_selectors=self.source_selectors,
            replay_indices=self.replay_indices,
            plan=self.plan,
            final_columns=self.final_columns,
            block_size=self.kv_cache_spec.block_size,
            max_tokens=self.num_spec + 1,
        )


def plan_gdn_recovery(
    *,
    num_accepted_tokens: torch.Tensor,
    request_indices: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_table: torch.Tensor,
    source_slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    replay_indices: torch.Tensor,
    plan: torch.Tensor,
    final_columns: torch.Tensor,
    block_size: int,
    max_tokens: int,
) -> None:
    n = source_slots.numel()
    if n:
        _plan_gdn_recovery[(n,)](
            num_accepted_tokens,
            request_indices,
            num_computed_tokens,
            block_table,
            source_slots,
            cu_seqlens,
            replay_indices,
            plan,
            final_columns,
            block_table.stride(0),
            source_slots.stride(0),
            BLOCK_SIZE=block_size,
            MAX_TOKENS=max_tokens,
        )


@triton.jit
def _plan_gdn_recovery(
    accepted_ptr,
    request_ptr,
    computed_ptr,
    table_ptr,
    source_ptr,
    cu_ptr,
    replay_ptr,
    plan_ptr,
    columns_ptr,
    table_stride: tl.constexpr,
    source_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    batch = tl.load(request_ptr + row)
    length = tl.load(cu_ptr + row + 1) - tl.load(cu_ptr + row)
    accepted = tl.minimum(tl.maximum(tl.load(accepted_ptr + batch), 0), length)
    pre = tl.load(computed_ptr + batch)
    source = tl.load(source_ptr + row * source_stride)
    valid = (accepted > 0) & (source > 0)
    new = pre + accepted
    final_col = (new - 1) // BLOCK_SIZE
    boundary = (pre // BLOCK_SIZE + 1) * BLOCK_SIZE
    has_prefix = valid & (boundary <= new)
    prefix_count = tl.where(has_prefix, boundary - pre, 0)
    prefix_col = boundary // BLOCK_SIZE - 1
    prefix_slot = tl.load(
        table_ptr + batch * table_stride + prefix_col, mask=has_prefix, other=0
    )
    final_slot = tl.load(
        table_ptr + batch * table_stride + final_col, mask=valid, other=0
    )
    tl.store(columns_ptr + row, tl.where(valid, final_col, -1))
    tl.store(plan_ptr + row * 5, source)
    tl.store(plan_ptr + row * 5 + 1, prefix_slot)
    tl.store(plan_ptr + row * 5 + 2, prefix_count)
    tl.store(plan_ptr + row * 5 + 3, final_slot)
    tl.store(plan_ptr + row * 5 + 4, tl.where(valid, accepted, 0))
    for t in range(MAX_TOKENS):
        slot = tl.where(has_prefix & (t + 1 == prefix_count), prefix_slot, 0)
        slot = tl.where(valid & (t + 1 == accepted), final_slot, slot)
        tl.store(replay_ptr + row * (MAX_TOKENS + 1) + t, slot)
    tl.store(replay_ptr + row * (MAX_TOKENS + 1) + MAX_TOKENS, source)


def compact_gdn_conv(
    conv_state: torch.Tensor, plan: torch.Tensor, *, history_size: int
) -> None:
    """Compact accepted and boundary windows, reading both before any alias write."""
    if plan.shape[0]:
        _compact_gdn_conv[(plan.shape[0], triton.cdiv(conv_state.shape[1], 128))](
            conv_state,
            plan,
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            CHANNELS=conv_state.shape[1],
            HISTORY=history_size,
            BLOCK_HISTORY=triton.next_power_of_2(history_size),
            BLOCK_CHANNELS=128,
        )


@triton.jit
def _compact_gdn_conv(
    state_ptr,
    plan_ptr,
    page_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    time_stride: tl.constexpr,
    CHANNELS: tl.constexpr,
    HISTORY: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    row = tl.program_id(0)
    source = tl.load(plan_ptr + row * 5).to(tl.int64)
    prefix = tl.load(plan_ptr + row * 5 + 1).to(tl.int64)
    prefix_count = tl.load(plan_ptr + row * 5 + 2)
    final = tl.load(plan_ptr + row * 5 + 3).to(tl.int64)
    accepted = tl.load(plan_ptr + row * 5 + 4)
    if accepted <= 0:
        return
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    history = tl.arange(0, BLOCK_HISTORY)
    offsets = channels[:, None] * channel_stride + history[None, :] * time_stride
    mask = (channels[:, None] < CHANNELS) & (history[None, :] < HISTORY)
    final_window = tl.load(
        state_ptr + source * page_stride + offsets + (accepted - 1) * time_stride,
        mask=mask,
        other=0,
    )
    prefix_window = tl.load(
        state_ptr + source * page_stride + offsets + (prefix_count - 1) * time_stride,
        mask=mask & (prefix_count > 0),
        other=0,
    )
    # Each channel belongs to one CTA; all source reads precede aliasing stores.
    tl.debug_barrier()
    tl.store(
        state_ptr + prefix * page_stride + offsets,
        prefix_window,
        mask=mask & (prefix_count > 0),
    )
    tl.store(state_ptr + final * page_stride + offsets, final_window, mask=mask)
