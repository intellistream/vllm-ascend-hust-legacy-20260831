# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    path = Path(__file__).parents[3] / "tools/benchmark_kv_gather_vs_span.py"
    spec = importlib.util.spec_from_file_location("benchmark_kv_gather_vs_span", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fragmented_mapping_produces_requested_spans():
    benchmark = _load_benchmark_module()

    pairs = benchmark.build_fragmented_mapping(
        selected_blocks=16,
        requested_span_len=4,
        num_cpu_blocks=128,
        num_npu_blocks=128,
        seed=7,
    )

    assert len(pairs) == 16
    assert len({cpu for cpu, _ in pairs}) == 16
    assert len({npu for _, npu in pairs}) == 16
    assert benchmark.coalesce_block_copy_spans(pairs) == [(*pairs[index], 4) for index in range(0, 16, 4)]


def test_partial_final_span_is_preserved():
    benchmark = _load_benchmark_module()

    case = benchmark.make_case(
        block_bytes=4096,
        selected_blocks=10,
        requested_span_len=4,
        num_cpu_blocks=128,
        num_npu_blocks=128,
        seed=11,
    )

    assert [length for _, _, length in case.spans] == [4, 4, 2]
    assert case.span_count == 3
    assert case.max_span_len == 4


def test_mapping_rejects_insufficient_address_space():
    benchmark = _load_benchmark_module()

    try:
        benchmark.build_fragmented_mapping(
            selected_blocks=16,
            requested_span_len=1,
            num_cpu_blocks=16,
            num_npu_blocks=16,
            seed=0,
        )
    except ValueError as exc:
        assert "not enough" in str(exc)
    else:
        raise AssertionError("expected fragmented mapping to reject insufficient blocks")
