import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK = (
    Path(__file__).parents[3]
    / "tools"
    / "benchmark_aclgraph_kv_restore_overlap.py"
)
SPEC = importlib.util.spec_from_file_location("aclgraph_overlap_benchmark", BENCHMARK)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(BENCHMARK.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("combined_ms", "expected"),
    [
        (10.0, "overlap_or_shorter"),
        (15.0, "serialized"),
        (12.0, "partial_overlap"),
        (17.0, "interference_or_overhead"),
    ],
)
def test_classify_timeline(combined_ms, expected):
    metrics = MODULE.calculate_overlap_metrics(10.0, 5.0, combined_ms)
    assert MODULE.classify_timeline(metrics, 0.1) == expected


def test_calculate_overlap_metrics_for_full_overlap():
    metrics = MODULE.calculate_overlap_metrics(8.0, 3.0, 8.0)
    assert metrics == {
        "serialized_ms": 11.0,
        "ideal_overlap_ms": 8.0,
        "hidden_ms": 3.0,
        "hidden_fraction": 1.0,
        "over_ideal_ms": 0.0,
        "over_serialized_ms": -3.0,
    }


def test_calculate_overlap_metrics_rejects_non_positive_components():
    with pytest.raises(ValueError, match="positive"):
        MODULE.calculate_overlap_metrics(0.0, 1.0, 1.0)


def test_remove_measurement_floor():
    assert MODULE.remove_measurement_floor(1.25, 0.25) == 1.0
    with pytest.raises(ValueError, match="does not exceed"):
        MODULE.remove_measurement_floor(0.25, 0.25)


def test_mode_list_contains_required_experiment_matrix():
    assert MODULE.MODE_NAMES == (
        "graph_only",
        "span_only",
        "mapped_only",
        "graph_overlap_span",
        "mapped_then_graph",
        "graph_overlap_mapped",
    )
