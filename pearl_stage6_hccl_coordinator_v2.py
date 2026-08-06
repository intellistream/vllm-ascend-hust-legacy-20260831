#!/usr/bin/env python3
"""Stage-6 coordinator with immediate HCCL-sidecar shutdown."""

from __future__ import annotations

import subprocess

import pearl_stage6_hccl_coordinator_v1 as base


def stop_hccl_sidecar(process: subprocess.Popen) -> None:
    """Terminate a sidecar immediately; it is expected to block in HCCL recv."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


base.stop_process = stop_hccl_sidecar


if __name__ == "__main__":
    base.main()
