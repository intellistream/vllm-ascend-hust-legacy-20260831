from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class _GraphCaptureOnlyBuilder:

    def __init__(self):
        self.called_with = None

    def build_for_graph_capture(self, common_attn_metadata):
        self.called_with = common_attn_metadata
        return "capture-metadata"


class _BaseCaptureOnlyBuilder:

    def __init__(self):
        self.called_with = None

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.called_with = common_attn_metadata
        return "base-capture-metadata"


class _BuildBuilder:

    def __init__(self):
        self.called_with = None
        self.common_prefix_len = None

    def build(self, *, common_prefix_len, common_attn_metadata):
        self.common_prefix_len = common_prefix_len
        self.called_with = common_attn_metadata
        return "build-metadata"


def _make_kv_cache_config():
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [object()]
    return kv_cache_config


def _build_metadata(builder, *, for_cudagraph_capture):
    attn_group = MagicMock()
    attn_group.get_metadata_builder.return_value = builder
    attn_group.layer_names = ["layer.0"]

    return build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=[torch.zeros((1, 1), dtype=torch.int32)],
        slot_mappings=[torch.zeros(1, dtype=torch.int64)],
        kv_cache_config=_make_kv_cache_config(),
        seq_lens_np=np.array([1], dtype=np.int32),
        for_cudagraph_capture=for_cudagraph_capture,
    )


def test_build_attn_metadata_uses_ascend_graph_capture_builder():
    builder = _GraphCaptureOnlyBuilder()

    metadata = _build_metadata(builder, for_cudagraph_capture=True)

    assert metadata == {"layer.0": "capture-metadata"}
    assert builder.called_with is not None


def test_build_attn_metadata_falls_back_to_base_cudagraph_builder():
    builder = _BaseCaptureOnlyBuilder()

    metadata = _build_metadata(builder, for_cudagraph_capture=True)

    assert metadata == {"layer.0": "base-capture-metadata"}
    assert builder.called_with is not None


def test_build_attn_metadata_uses_regular_builder_outside_capture():
    builder = _BuildBuilder()

    metadata = _build_metadata(builder, for_cudagraph_capture=False)

    assert metadata == {"layer.0": "build-metadata"}
    assert builder.common_prefix_len == 0
    assert builder.called_with is not None
