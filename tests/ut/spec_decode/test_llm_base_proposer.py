# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class _RecordingMetadataBuilder:
    def __init__(self):
        self.kwargs = None

    def build_for_drafting(self, _common_attn_metadata, draft_index, **kwargs):
        self.draft_index = draft_index
        self.kwargs = kwargs
        return SimpleNamespace(causal=True)


def test_dspark_draft_metadata_uses_non_compressed_group_block_size():
    builder = _RecordingMetadataBuilder()
    attn_group = SimpleNamespace(
        get_metadata_builder=lambda: builder,
        kv_cache_group_id="swa",
        kv_cache_spec=SimpleNamespace(block_size=64),
        layer_names=["draft.attn"],
    )
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.draft_attn_groups = [attn_group]
    proposer.method = "dspark"
    proposer.use_compress = False
    proposer._per_group_block_table_buffers = {}
    proposer._per_group_query_slot_mapping_buffers = {"swa": None}

    proposer.build_draft_attn_metadata(SimpleNamespace(num_input_tokens=0), 0, 0)

    assert builder.draft_index == 1
    assert builder.kwargs == {"block_size": 64}
