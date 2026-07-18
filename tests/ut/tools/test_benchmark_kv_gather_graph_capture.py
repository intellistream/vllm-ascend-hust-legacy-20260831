# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_benchmark_module():
    path = Path(__file__).parents[3] / "tools/benchmark_kv_gather_graph_capture.py"
    spec = importlib.util.spec_from_file_location("benchmark_kv_gather_graph_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_ids_keep_shape_but_change_contents():
    benchmark = _load_benchmark_module()
    first = benchmark.build_source_ids(num_cpu_blocks=64, selected_blocks=16, seed=7, replay_variant=0)
    second = benchmark.build_source_ids(num_cpu_blocks=64, selected_blocks=16, seed=7, replay_variant=1)

    assert len(first) == len(second) == 16
    assert len(set(first)) == len(set(second)) == 16
    assert first != second
    assert all(0 <= block < 64 for block in first + second)


def test_shape_validation_returns_elements_per_block():
    benchmark = _load_benchmark_module()
    assert (
        benchmark.validate_shape_config(
            block_bytes=4096,
            element_size=2,
            selected_blocks=8,
            num_cpu_blocks=32,
            num_npu_blocks=16,
            decode_rows=4,
            surrogate_hidden=512,
            surrogate_depth=2,
        )
        == 2048
    )


@pytest.mark.parametrize(
    "override",
    [
        {"selected_blocks": 17},
        {"decode_rows": 9},
        {"surrogate_hidden": 4096},
        {"surrogate_depth": 0},
    ],
)
def test_shape_validation_rejects_invalid_graph_shapes(override):
    benchmark = _load_benchmark_module()
    kwargs = {
        "block_bytes": 4096,
        "element_size": 2,
        "selected_blocks": 8,
        "num_cpu_blocks": 32,
        "num_npu_blocks": 16,
        "decode_rows": 8,
        "surrogate_hidden": 512,
        "surrogate_depth": 2,
    }
    kwargs.update(override)
    with pytest.raises(ValueError):
        benchmark.validate_shape_config(**kwargs)


def test_comparison_uses_graph_only_as_exposed_cost_floor():
    benchmark = _load_benchmark_module()
    comparison = benchmark.derive_comparison(
        {
            "graph_only": {"mean_ms": 2.0},
            "mapped_then_graph": {"mean_ms": 3.0},
            "graph_capture_gather_decode": {"mean_ms": 2.5},
        }
    )

    assert comparison == {
        "outside_exposed_over_graph_ms": 1.0,
        "captured_exposed_over_graph_ms": 0.5,
        "capture_savings_ms": 0.5,
        "capture_savings_percent": pytest.approx(100.0 / 6.0),
    }


def test_operator_provenance_hashes_full_vendor_package(tmp_path):
    benchmark = _load_benchmark_module()
    vendor = tmp_path / "vendors" / "custom_transformer"
    opapi = vendor / "op_api" / "lib" / "libcust_opapi.so"
    source = vendor / (
        "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/kv_cache_block_gather/kv_cache_block_gather.h"
    )
    kernel = vendor / ("op_impl/ai_core/tbe/kernel/ascend910b/kv_cache_block_gather/kernel.o")
    for path, contents in (
        (opapi, b"opapi"),
        (source, b"two queues"),
        (kernel, b"kernel"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    provenance = benchmark.collect_operator_provenance(opapi)

    assert provenance["vendor_root"] == str(vendor.resolve())
    assert provenance["kernel_source_sha256"] == benchmark.file_sha256(source)
    assert provenance["kernel_objects_sha256"] == {str(kernel.relative_to(vendor)): benchmark.file_sha256(kernel)}
