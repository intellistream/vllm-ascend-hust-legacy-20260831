#!/usr/bin/env python3
"""Resolve Ascend benchmark scenario for the vllm-hust CI pipeline.

Called by the external vllm-hust-benchmark runner to determine which
benchmark scenario to execute.  Outputs the resolved scenario name on stdout.
"""
import os
import sys


def resolve_benchmark_scenario() -> str:
    return os.environ.get("BENCH_SCENARIO", "random-online")


if __name__ == "__main__":
    scenario = resolve_benchmark_scenario()
    sys.stdout.write(scenario)
