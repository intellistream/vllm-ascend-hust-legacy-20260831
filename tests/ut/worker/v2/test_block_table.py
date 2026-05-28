from unittest.mock import patch

import torch

from vllm_ascend.worker.v2.block_table import AscendBlockTables


class _FakeKernel:

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def test_compute_slot_mappings_limits_padding_to_returned_slice():
    block_tables = AscendBlockTables(
        block_sizes=[128],
        max_num_reqs=4,
        max_num_batched_tokens=128,
        max_model_len=4096,
        device=torch.device("cpu"),
    )
    fake_kernel = _FakeKernel()

    idx_mapping = torch.tensor([0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    positions = torch.arange(5, dtype=torch.int64)

    with patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel", fake_kernel):
        slot_mappings = block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded=8,
        )

    assert slot_mappings.shape == (1, 8)
    assert len(fake_kernel.calls) == 1
    _, args, _ = fake_kernel.calls[0]
    assert args[0] == 8
