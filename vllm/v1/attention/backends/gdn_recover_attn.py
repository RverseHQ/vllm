# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact speculative GDN replay with one physical running-state slot."""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
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
    from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


def _device_decode_metadata_enabled(vllm_config: "VllmConfig") -> bool:
    spec = vllm_config.speculative_config
    return envs.VLLM_GDN_DECODE_KERNEL.lower() == "cuda" and (
        envs.VLLM_QWEN_GDN_DEVICE_METADATA
        or (spec is not None and spec.enable_adaptive_verification)
    )


class GDNRecoverAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVER_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNRecoverAttentionMetadataBuilder"]:
        return GDNRecoverAttentionMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Recovery plans read device cu_seqlens, not CPU per-request lengths.
        return envs.VLLM_GDN_DECODE_KERNEL.lower() == "cuda"


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
    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls, vllm_config: "VllmConfig", kv_cache_spec: "KVCacheSpec"
    ) -> int | None:
        if (
            _device_decode_metadata_enabled(vllm_config)
            and vllm_config.num_speculative_tokens == 3
        ):
            return 4
        return None

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
        self.decode_request_indices = (
            torch.empty(n, dtype=torch.int32, device=device)
            if _device_decode_metadata_enabled(vllm_config)
            else None
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    ) -> GDNRecoverAttentionMetadata:
        metadata = self._build_device_decode_metadata(
            common_attn_metadata, num_decode_draft_tokens_cpu
        )
        if metadata is None:
            metadata = super().build(
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                fast_build=fast_build,
            )
            spec = self.vllm_config.speculative_config
            if spec is not None and spec.enable_adaptive_verification:
                metadata.uniform_spec_sequence_length = None
            n = metadata.num_spec_decodes
            request_indices_cpu = torch.empty(0, dtype=torch.int32)
            if n:
                assert num_decode_draft_tokens_cpu is not None
                request_indices_cpu = torch.nonzero(
                    num_decode_draft_tokens_cpu >= 0, as_tuple=True
                )[0].to(torch.int32)
                assert metadata.spec_state_indices_tensor is not None
                assert metadata.num_accepted_tokens is not None
                self.verify_indices[:n, -1].copy_(
                    metadata.spec_state_indices_tensor[:n, 0]
                )
                self.verify_indices[n:, -1].zero_()
                # Recovery compacted convolution history to offset zero.
                metadata.num_accepted_tokens.fill_(1)
            request_indices = async_tensor_h2d(
                request_indices_cpu, device=self.verify_indices.device
            )
        else:
            assert self.decode_request_indices is not None
            request_indices = self.decode_request_indices[: metadata.num_spec_decodes]
        return GDNRecoverAttentionMetadata(
            **vars(metadata),
            layers=self.layers,
            recovery_batch=self.recovery_batch,
            block_table=common_attn_metadata.block_table_tensor,
            num_computed_tokens=common_attn_metadata.compute_num_computed_tokens(),
            request_indices=request_indices,
            verify_indices=self.verify_indices,
            source_selectors=self.source_selectors,
            replay_indices=self.replay_indices,
            plan=self.plan,
            final_columns=self.final_columns,
            block_size=self.kv_cache_spec.block_size,
            max_tokens=self.num_spec + 1,
        )

    def _build_device_decode_metadata(
        self,
        common: CommonAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata | None:
        if (
            self.decode_request_indices is None
            or not common.query_start_loc.is_cuda
            or num_decode_draft_tokens_cpu is None
            or common.max_query_len > self.num_spec + 1
            or common.num_actual_tokens > self.decode_cudagraph_max_bs
        ):
            return None
        spec_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = int(spec_masks_cpu.sum())
        if not num_spec_decodes or not bool(spec_masks_cpu[:num_spec_decodes].all()):
            return None
        # Non-spec rows may only be trailing CUDA-graph padding.
        query_cpu = common.query_start_loc_cpu
        if bool((query_cpu[num_spec_decodes:] != query_cpu[-1]).any()):
            return None
        num_tokens = int(query_cpu[-1])
        batch_size = common.num_reqs
        prepare_gdn_decode_metadata(
            block_table=common.block_table_tensor,
            seq_lens=common.seq_lens,
            query_start_loc=common.query_start_loc,
            state_indices=self.spec_state_indices_tensor[:batch_size],
            verify_indices=self.verify_indices[:batch_size],
            spec_masks=self.spec_sequence_masks[:batch_size],
            accepted=self.num_accepted_tokens[:batch_size],
            request_indices=self.decode_request_indices[:batch_size],
            query_start_loc_out=self.spec_query_start_loc[: batch_size + 1],
            token_indices=self.spec_token_indx[:num_tokens],
            num_spec_decodes=num_spec_decodes,
            block_size=self.kv_cache_spec.block_size,
        )
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_tokens,
            num_actual_tokens=common.num_actual_tokens,
            has_initial_state=None,
            spec_query_start_loc=self.spec_query_start_loc[: batch_size + 1],
            non_spec_query_start_loc=None,
            spec_state_indices_tensor=self.spec_state_indices_tensor[:batch_size],
            non_spec_state_indices_tensor=None,
            spec_sequence_masks=self.spec_sequence_masks[:batch_size],
            spec_sequence_masks_cpu=spec_masks_cpu,
            spec_token_indx=self.spec_token_indx[:num_tokens],
            non_spec_token_indx=self.non_spec_token_indx[:0],
            num_accepted_tokens=self.num_accepted_tokens[:batch_size],
            # The device split may differ from the CPU's even allocation.
            uniform_spec_sequence_length=None,
        )


def prepare_gdn_decode_metadata(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    verify_indices: torch.Tensor,
    spec_masks: torch.Tensor,
    accepted: torch.Tensor,
    request_indices: torch.Tensor,
    query_start_loc_out: torch.Tensor,
    token_indices: torch.Tensor,
    num_spec_decodes: int,
    block_size: int,
) -> None:
    """Stage aligned decode slots and live device query boundaries in one launch."""
    num_reqs = seq_lens.numel()
    num_tokens = token_indices.numel()
    _prepare_gdn_decode_metadata[(triton.cdiv(max(num_reqs + 1, num_tokens), 128),)](
        block_table,
        seq_lens,
        query_start_loc,
        state_indices,
        verify_indices,
        spec_masks,
        accepted,
        request_indices,
        query_start_loc_out,
        token_indices,
        num_reqs,
        num_spec_decodes,
        num_tokens,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        state_indices.stride(0),
        verify_indices.stride(0),
        BLOCK_SIZE=block_size,
        VERIFY_WIDTH=verify_indices.shape[1],
        VERIFY_BLOCK=triton.next_power_of_2(verify_indices.shape[1]),
        BLOCK=128,
    )


@triton.jit
def _prepare_gdn_decode_metadata(
    table_ptr,
    seq_ptr,
    query_ptr,
    state_ptr,
    verify_ptr,
    masks_ptr,
    accepted_ptr,
    request_ptr,
    query_out_ptr,
    token_ptr,
    num_reqs,
    num_spec_decodes,
    num_tokens,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    seq_stride: tl.constexpr,
    state_row_stride: tl.constexpr,
    verify_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VERIFY_WIDTH: tl.constexpr,
    VERIFY_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < num_spec_decodes
    sequence = tl.load(seq_ptr + row * seq_stride, live, other=0)
    column = tl.maximum((sequence.to(tl.int64) - 1) // BLOCK_SIZE, 0)
    source = tl.load(
        table_ptr + row.to(tl.int64) * table_row_stride + column * table_col_stride,
        live,
        other=0,
    )
    tl.store(state_ptr + row * state_row_stride, source, row < num_reqs)
    verify_col = tl.arange(0, VERIFY_BLOCK)
    tl.store(
        verify_ptr + row[:, None] * verify_row_stride + verify_col[None, :],
        tl.where(verify_col[None, :] == VERIFY_WIDTH - 1, source[:, None], 0),
        (row[:, None] < num_reqs) & (verify_col[None, :] < VERIFY_WIDTH),
    )
    tl.store(masks_ptr + row, live, row < num_reqs)
    tl.store(accepted_ptr + row, 1, row < num_reqs)
    tl.store(request_ptr + row, row, row < num_reqs)
    query = tl.load(query_ptr + row, row <= num_reqs, other=0)
    tl.store(query_out_ptr + row, query, row <= num_reqs)
    tl.store(token_ptr + row, row, row < num_tokens)


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
