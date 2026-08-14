# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.models.deepseek_v4 import _store_mtp_hidden_states_impl


def test_store_mtp_hidden_states_only_updates_live_rows() -> None:
    buffer = torch.full((8, 4), -1.0)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    _store_mtp_hidden_states_impl(buffer, hidden_states)

    torch.testing.assert_close(buffer[:3], hidden_states)
    torch.testing.assert_close(buffer[3:], torch.full((5, 4), -1.0))
