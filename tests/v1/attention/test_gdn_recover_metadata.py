# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device query boundaries must survive GDN recovery metadata and graph replay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointBuilder
from vllm.model_executor.layers.mamba.ops.gdn_recover import GdnRecoverBatch
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_recover_attn import (
    GDNRecoverAttentionBackend,
    GDNRecoverAttentionMetadataBuilder,
    prepare_gdn_decode_metadata,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import (
    get_query_lens_mismatch_unsupported_backend,
    get_varlen_cudagraph_unsupported_backend,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _decode_step():
    # Only allocate the metadata resources: no model weights or native replay
    # descriptors are needed to execute the real builder and staging kernel.
    builder = GDNRecoverAttentionMetadataBuilder.__new__(
        GDNRecoverAttentionMetadataBuilder
    )
    builder.num_spec = 3
    builder.decode_cudagraph_max_bs = 192
    builder.kv_cache_spec = MambaSpec(
        block_size=1600,
        shapes=((6, 10240), (48, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        num_speculative_blocks=0,
    )
    builder.layers = ()
    builder.recovery_batch = GdnRecoverBatch(
        torch.empty((0, 16), dtype=torch.int64, device="cuda"),
        torch.empty((0, 16), dtype=torch.int64),
    )
    builder.verify_indices = torch.full((6, 5), 77, dtype=torch.int32, device="cuda")
    builder.source_selectors = torch.full((6,), 5, dtype=torch.int32, device="cuda")
    builder.replay_indices = torch.empty_like(builder.verify_indices)
    builder.plan = torch.empty((6, 5), dtype=torch.int32, device="cuda")
    builder.final_columns = torch.empty(6, dtype=torch.int32, device="cuda")
    builder.decode_request_indices = torch.empty(6, dtype=torch.int32, device="cuda")
    builder.spec_state_indices_tensor = torch.empty(
        (6, 1), dtype=torch.int32, device="cuda"
    )
    builder.spec_sequence_masks = torch.empty(6, dtype=torch.bool, device="cuda")
    builder.num_accepted_tokens = torch.empty(6, dtype=torch.int32, device="cuda")
    builder.spec_query_start_loc = torch.empty(7, dtype=torch.int32, device="cuda")
    builder.spec_token_indx = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    builder.non_spec_token_indx = torch.empty(0, dtype=torch.int32, device="cuda")
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor(
            [0, 4, 5, 7, 8, 8, 8], device="cuda", dtype=torch.int32
        ),
        # AV knows the total on the host, but allocates per-request lengths on device.
        query_start_loc_cpu=torch.tensor([0, 2, 4, 6, 8, 8, 8], dtype=torch.int32),
        seq_lens=torch.tensor(
            [1601, 3199, 3200, 6401, 0, 0], device="cuda", dtype=torch.int32
        ),
        num_reqs=6,
        num_actual_tokens=8,
        max_query_len=4,
        max_seq_len=6401,
        block_table_tensor=torch.arange(1, 49, device="cuda", dtype=torch.int32).view(
            6, 8
        ),
        slot_mapping=torch.empty(8, device="cuda", dtype=torch.int64),
    )
    return builder, common


@pytest.mark.parametrize(
    "mode,capture_size,expected_capacity",
    [("NONE", 0, 48), ("FULL_AND_PIECEWISE", 192, 192)],
)
def test_recovery_metadata_covers_real_request_capacity(
    mode, capture_size, expected_capacity
):
    from tests.v1.attention.utils import (
        BatchSpec,
        create_common_attn_metadata,
        create_vllm_config,
    )
    from vllm.config import CUDAGraphMode, SpeculativeConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.ops.gdn_recover import GdnRecoverBatchStorage

    device = torch.device("cuda")
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=1600,
        max_model_len=4096,
        max_num_seqs=48,
    )
    config.additional_config = {"gdn_prefill_backend": "triton"}
    config.compilation_config.cudagraph_mode = CUDAGraphMode[mode]
    config.compilation_config.max_cudagraph_capture_size = capture_size
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    prefix = "model.layers.0.linear_attn"
    layer = object.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.gdn_recovery_storage = GdnRecoverBatchStorage(device)
    config.compilation_config.static_forward_context[prefix] = layer
    spec = MambaSpec(
        block_size=1600,
        shapes=((6, 10240), (48, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        num_speculative_blocks=0,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[8] * 48, query_lens=[4] * 48),
        1600,
        device,
        arange_block_indices=True,
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(config):
        builder = GDNRecoverAttentionMetadataBuilder(spec, [prefix], config, device)
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.ones(48, device=device, dtype=torch.int32),
            num_decode_draft_tokens_cpu=torch.full((48,), 3, dtype=torch.int32),
        )
    assert metadata.num_spec_decodes == 48
    assert metadata.verify_indices.shape == (expected_capacity, 5)
    assert metadata.verify_indices[:48, -1].tolist() == list(range(1, 49))
    assert metadata.source_selectors[:48].tolist() == [5] * 48


def test_recovery_builder_uses_device_lengths_and_clears_padding():
    builder, common = _decode_step()
    metadata = builder.build(
        0,
        common,
        num_decode_draft_tokens_cpu=torch.tensor([1, 1, 1, 1, -1, -1]),
    )
    assert metadata.num_spec_decodes == 4
    assert metadata.num_spec_decode_tokens == 8
    assert metadata.uniform_spec_sequence_length is None
    assert metadata.spec_query_start_loc.tolist() == [0, 4, 5, 7, 8, 8, 8]
    assert metadata.spec_state_indices_tensor[:, 0].tolist() == [2, 10, 18, 29, 0, 0]
    assert metadata.verify_indices[:, :4].count_nonzero().item() == 0
    assert metadata.verify_indices[:, 4].tolist() == [2, 10, 18, 29, 0, 0]
    assert metadata.spec_sequence_masks.tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert metadata.num_accepted_tokens.tolist() == [1] * 6
    assert metadata.request_indices.tolist() == [0, 1, 2, 3]
    assert metadata.spec_token_indx.tolist() == list(range(8))
    assert metadata.num_computed_tokens.tolist() == [1597, 3198, 3198, 6400, 0, 0]


def test_mixed_prefill_retains_device_spec_boundaries_without_uniform_hint():
    builder, common = _decode_step()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
    )
    builder.use_spec_decode = True
    builder.use_full_cuda_graph = True
    builder.gdn_prefill_backend = "flashinfer"
    builder.plan_flashinfer_checkpoints = False
    builder.checkpoint_builder = MambaPrefillCheckpointBuilder(
        builder.vllm_config, builder.kv_cache_spec
    )
    common = common.replace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 2, 4, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 4, 8], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([1601, 3199, 4], dtype=torch.int32, device="cuda"),
        seq_lens_cpu_upper_bound=torch.tensor([1601, 3199, 4], dtype=torch.int32),
        block_table_tensor=common.block_table_tensor[:3],
    )
    metadata = builder.build(
        0,
        common,
        num_decode_draft_tokens_cpu=torch.tensor([1, 1, -1]),
        num_accepted_tokens=torch.ones(3, dtype=torch.int32, device="cuda"),
    )
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == 4
    assert metadata.num_spec_decodes == 2
    assert metadata.spec_query_start_loc.tolist() == [0, 3, 4]
    assert metadata.uniform_spec_sequence_length is None


def test_device_decode_metadata_replays_new_lengths_and_slots():
    builder, common = _decode_step()

    def stage():
        prepare_gdn_decode_metadata(
            block_table=common.block_table_tensor,
            seq_lens=common.seq_lens,
            query_start_loc=common.query_start_loc,
            state_indices=builder.spec_state_indices_tensor,
            verify_indices=builder.verify_indices,
            spec_masks=builder.spec_sequence_masks,
            accepted=builder.num_accepted_tokens,
            request_indices=builder.decode_request_indices,
            query_start_loc_out=builder.spec_query_start_loc,
            token_indices=builder.spec_token_indx,
            num_spec_decodes=4,
            block_size=1600,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stage()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        stage()
    for queries, lengths, slots in [
        ([0, 1, 5, 6, 8, 8, 8], [1600, 3201, 4801, 6400, 0, 0], [1, 11, 20, 28, 0, 0]),
        ([0, 2, 4, 7, 8, 8, 8], [1601, 3199, 3200, 6401, 0, 0], [2, 10, 18, 29, 0, 0]),
    ]:
        common.query_start_loc.copy_(
            torch.tensor(queries, dtype=torch.int32, device="cuda")
        )
        common.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        graph.replay()
        assert builder.spec_query_start_loc.tolist() == queries
        assert builder.spec_state_indices_tensor[:, 0].tolist() == slots
        assert builder.verify_indices[:, 4].tolist() == slots
        assert builder.verify_indices[:, :4].count_nonzero().item() == 0


def test_adaptive_verification_accepts_recover_gdn_varlen_graph(monkeypatch):
    monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "CUDA")
    monkeypatch.setenv("VLLM_QWEN_GDN_DEVICE_METADATA", "0")
    builder, _ = _decode_step()
    config = SimpleNamespace(
        num_speculative_tokens=3,
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
    )
    group = SimpleNamespace(
        layer_names=["gdn"],
        kv_cache_spec=builder.kv_cache_spec,
        backend=GDNRecoverAttentionBackend,
        get_metadata_builder=lambda _: builder,
    )
    assert get_query_lens_mismatch_unsupported_backend([[group]]) is None
    assert get_varlen_cudagraph_unsupported_backend([[group]], config, 4) is None
    assert get_varlen_cudagraph_unsupported_backend([[group]], config, 5) == (
        "GDNRecoverAttentionBackend",
        4,
    )
