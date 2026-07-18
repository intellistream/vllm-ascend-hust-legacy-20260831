import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "tools" / "benchmark_hybrid_kv_attention_promotion.py"
SPEC = importlib.util.spec_from_file_location("hybrid_promotion_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("fraction, expected_host", [(0.0, 0), (0.25, 4),
                                                       (0.5, 8), (1.0, 16)])
def test_build_layout_counts_and_dense_source_ids(fraction, expected_host):
    layout = MODULE.build_layout(16, fraction)
    assert layout.host_blocks == expected_host
    assert layout.device_blocks == 16 - expected_host
    assert len(layout.source_kinds) == len(layout.source_block_ids) == 16
    host_ids = [block_id for kind, block_id in zip(layout.source_kinds,
                                                    layout.source_block_ids) if kind]
    device_ids = [block_id for kind, block_id in zip(layout.source_kinds,
                                                      layout.source_block_ids) if not kind]
    assert host_ids == list(range(expected_host))
    assert device_ids == list(range(16 - expected_host))


@pytest.mark.parametrize("blocks, fraction", [(0, 0.5), (4, -0.1), (4, 1.1)])
def test_build_layout_rejects_invalid_inputs(blocks, fraction):
    with pytest.raises(ValueError):
        MODULE.build_layout(blocks, fraction)


def test_summarize_crossovers_reports_penalty_and_first_win():
    results = []
    for tokens, device, permanent, promotion in (
        (1, 1.0, 2.0, 2.5),
        (2, 2.0, 4.0, 3.5),
        (4, 4.0, 8.0, 5.5),
    ):
        for policy, mean_ms in (("device", device),
                                ("permanent_hybrid", permanent),
                                ("promote_first_use", promotion)):
            results.append({"block_elems": 1024, "selected_blocks": 16,
                            "host_fraction": 1.0, "tokens": tokens,
                            "policy": policy, "mean_ms": mean_ms})
    summary = MODULE.summarize_crossovers(results)[0]
    assert summary["promotion_crossover_tokens"] == 2
    assert summary["comparisons"][0]["permanent_penalty_vs_device_percent"] == 100.0
    assert summary["comparisons"][1]["promotion_speedup_vs_permanent_percent"] == 12.5
