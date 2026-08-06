#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

repo = Path(__file__).resolve().parent
coordinator = repo / "pearl_stage5_coordinator.py"

sys.argv[0] = str(coordinator)
sys.argv.extend([
    "--transport", "hccl",
    "--hccl-master-addr", "127.0.0.1",
    "--hccl-master-port", "29646",
])

runpy.run_path(str(coordinator), run_name="__main__")
