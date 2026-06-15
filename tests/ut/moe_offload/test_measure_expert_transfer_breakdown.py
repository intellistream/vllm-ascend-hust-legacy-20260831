# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.sew_offload.measure_expert_transfer_breakdown import (
    CopyPatternSpec,
    infer_expert_shape_from_config,
    linear_fit,
    make_breakdown,
    parse_size_factors,
    summarize_api_statistic_csv,
    summarize_pcie_csv,
    summarize_trace_copy_patterns,
)


def test_parse_size_factors_requires_multiple_positive_values():
    assert parse_size_factors("0.5,1,2") == (0.5, 1.0, 2.0)

    with pytest.raises(ValueError, match="at least two"):
        parse_size_factors("1")
    with pytest.raises(ValueError, match="positive"):
        parse_size_factors("1,0")


def test_infer_expert_shape_from_qwen3_moe_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "hidden_size": 2048,
                "moe_intermediate_size": 768,
                "intermediate_size": 6144,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    shape = infer_expert_shape_from_config(config_path)

    assert shape.w13_shape == (1536, 2048)
    assert shape.w2_shape == (2048, 768)
    assert shape.dtype_name == "bfloat16"
    assert (shape.w13_elements + shape.w2_elements) * 2 == 9_437_184


def test_linear_fit_recovers_payload_slope_and_fixed_overhead():
    fit = linear_fit(
        [
            (1_000_000, 0.15),
            (2_000_000, 0.25),
            (4_000_000, 0.45),
        ]
    )

    assert fit.intercept_ms == pytest.approx(0.05)
    assert fit.slope_ms_per_byte == pytest.approx(1e-7)
    assert fit.payload_bandwidth_gbps == pytest.approx(10.0)


def test_make_breakdown_reports_payload_fixed_and_pcie_utilization():
    fit = linear_fit([(1_000_000, 0.15), (2_000_000, 0.25), (4_000_000, 0.45)])

    breakdown = make_breakdown(
        expert_bytes=2_000_000,
        expert_event_ms=0.30,
        expert_wall_ms=0.40,
        fit=fit,
        pcie_peak_gbps=20.0,
    )

    assert breakdown["payload_movement_ms_from_fit"] == pytest.approx(0.2)
    assert breakdown["fixed_plus_residual_ms"] == pytest.approx(0.1)
    assert breakdown["payload_fraction_of_event"] == pytest.approx(2 / 3)
    assert breakdown["effective_bandwidth_gbps_including_overhead"] == pytest.approx(6.6666667)
    assert breakdown["effective_pcie_utilization"] == pytest.approx(1 / 3)
    assert breakdown["payload_pcie_utilization"] == pytest.approx(0.5)


def test_summarize_trace_copy_patterns_groups_acl_memcpy_by_window(tmp_path):
    trace_path = tmp_path / "trace_view.json"
    trace_path.write_text(
        json.dumps(
            [
                {"ph": "X", "name": "sew_transfer_single", "ts": "1000", "dur": 1000},
                {"ph": "X", "name": "aten::copy_", "ts": "1050", "dur": 250},
                {"ph": "X", "name": "AscendCL@aclrtMemcpy", "ts": "1100", "dur": 200},
                {"ph": "X", "name": "AscendCL@aclrtSynchronizeStream", "ts": "1350", "dur": 30},
                {"ph": "C", "name": "PCIe_cpl", "ts": "1200", "args": {"Rx": 64000, "Tx": 0}},
                {"ph": "X", "name": "sew_transfer_two", "ts": "3000", "dur": 2000},
                {"ph": "X", "name": "aten::copy_", "ts": "3050", "dur": 350},
                {"ph": "X", "name": "AscendCL@aclrtMemcpy", "ts": "3100", "dur": 300},
                {"ph": "X", "name": "AscendCL@aclrtMemcpy", "ts": "3500", "dur": 400},
                {"ph": "X", "name": "AscendCL@aclrtMemcpy", "ts": "6000", "dur": 999},
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_trace_copy_patterns(
        trace_path=trace_path,
        pattern_specs=[
            CopyPatternSpec(
                name="single",
                bytes_per_iteration=1_000_000,
                copy_calls_per_iteration=1,
                repeats=1,
            ),
            CopyPatternSpec(
                name="two",
                bytes_per_iteration=2_000_000,
                copy_calls_per_iteration=2,
                repeats=1,
            ),
        ],
        pcie_peak_gbps=64.0,
    )

    single = summary["single"]
    assert single["record_window_count"] == 1
    assert single["aclrt_memcpy_count"] == 1
    assert single["aclrt_memcpy_us"] == pytest.approx(200)
    assert single["host_window_non_memcpy_us"] == pytest.approx(800)
    assert single["aclrt_memcpy_bandwidth_gbps"] == pytest.approx(5.0)
    assert single["pcie_trace_counters"]["PCIe_cpl"]["Rx"]["max_gbps"] == pytest.approx(64.0)

    two = summary["two"]
    assert two["aclrt_memcpy_count"] == 2
    assert two["aclrt_memcpy_expected_count_delta"] == 0
    assert two["aclrt_memcpy_us"] == pytest.approx(700)
    assert two["aclrt_memcpy_us_per_call"] == pytest.approx(350)


def test_summarize_profiler_csv_outputs(tmp_path):
    pcie_path = tmp_path / "pcie.csv"
    pcie_path.write_text(
        "\n".join(
            [
                "Device_id,Mode,Min,Max,Avg",
                "1,Rx_cpl_avg(MB/s),0.0,64000.0,32000.0",
                "1,Tx_latency_avg(us),0.5,1.5,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    api_path = tmp_path / "api_statistic.csv"
    api_path.write_text(
        "\n".join(
            [
                "Device_id,Level,API Name,Time(us),Count,Avg(us),Min(us),Max(us),Variance",
                "host,acl,aclrtMemcpy,900.0,3,300.0,250.0,350.0,100.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pcie_summary = summarize_pcie_csv(pcie_path, pcie_peak_gbps=64.0)
    assert pcie_summary["Rx_cpl_avg(MB/s)"]["max_gbps"] == pytest.approx(64.0)
    assert pcie_summary["Rx_cpl_avg(MB/s)"]["avg_pcie_utilization"] == pytest.approx(0.5)
    assert "max_gbps" not in pcie_summary["Tx_latency_avg(us)"]

    api_summary = summarize_api_statistic_csv(api_path)
    assert api_summary["aclrtMemcpy"]["count"] == 3
    assert api_summary["aclrtMemcpy"]["avg_us"] == pytest.approx(300.0)
