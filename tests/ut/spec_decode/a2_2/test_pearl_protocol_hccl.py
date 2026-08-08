# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-card HCCL coverage for the experimental PEARL transport protocol."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="PEARL HCCL smoke test requires two NPUs.")
def test_pearl_protocol_hccl_smoke() -> None:
    env = os.environ.copy()
    # Shared CI hosts can already have the default HCCL port in use.
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "vllm_ascend.spec_decode.pearl.smoke",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PEARL HCCL proposal and verification protocol passed." in result.stdout
