"""Dependency-light contract tests for inplace split containment.

This module deliberately loads the pure planner file without importing the
``vllm_ascend`` package, so it can run on a host without vLLM or an NPU stack.
"""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_utils():
    path = Path(__file__).parents[3] / "vllm_ascend/worker/inplace_split_utils.py"
    module_name = "_inplace_split_utils_host"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


utils = _load_utils()


class InplaceSplitContractsHostTest(unittest.TestCase):
    def test_enabled_config_rejects_unimplemented_modes_and_features(self):
        common = dict(
            enabled=True,
            mode="inplace_parallel",
            num_splits=2,
            enable_parallel_streams=True,
            enable_inplace_spec_decode=False,
            enable_inplace_mrope=False,
            replay_policy="full_graph_parallel",
        )
        utils.validate_inplace_split_config_values(**common)

        for override in (
            {"mode": "inplace_serial"},
            {"mode": "parallel_buffer"},
            {"num_splits": 3},
            {"enable_parallel_streams": False},
            {"enable_inplace_spec_decode": True},
            {"enable_inplace_mrope": True},
            {"replay_policy": "piecewise_attention_parallel"},
        ):
            values = common | override
            with self.subTest(override=override), self.assertRaises(ValueError):
                utils.validate_inplace_split_config_values(**values)

    def test_planner_rejects_inconsistent_or_nonuniform_schedule(self):
        for schedule, total in (([2, 1, 1], 4), ([1, 1, 1], 4)):
            plan, reason = utils.create_inplace_split_batch_slices(np.asarray(schedule), total, 1, [2, 8])
            self.assertIsNone(plan)
            self.assertEqual(reason, utils.NO_SPLIT_INVALID_SCHEDULE)

    def test_runtime_precheck_rejects_unsafe_topologies(self):
        common = dict(
            enabled=True,
            mode="inplace_parallel",
            parallel_streams=True,
            full_graph=True,
            uniform_decode=True,
            spec_decode=False,
            dp_world_size=1,
            has_gdn_or_hybrid=False,
            sp_enabled=False,
            context_parallel_enabled=False,
        )
        expected = (
            ({"full_graph": False}, utils.NO_SPLIT_CUDAGRAPH_MODE_NOT_FULL),
            ({"spec_decode": True}, utils.NO_SPLIT_SPEC_DECODE_CONFLICT),
            ({"dp_world_size": 2}, utils.NO_SPLIT_DP_UNSUPPORTED),
            ({"has_gdn_or_hybrid": True}, utils.NO_SPLIT_GDN_HYBRID_UNSUPPORTED),
            ({"sp_enabled": True}, utils.NO_SPLIT_SP_UNSUPPORTED),
            (
                {"context_parallel_enabled": True},
                utils.NO_SPLIT_CONTEXT_PARALLEL_UNSUPPORTED,
            ),
        )
        for override, reason in expected:
            with self.subTest(override=override):
                self.assertEqual(
                    utils.inplace_split_runtime_rejection(**(common | override)),
                    reason,
                )

    def test_planner_rejects_tp_incompatible_graph_shape(self):
        plan, reason = utils.create_inplace_split_batch_slices(
            np.ones(5, dtype=np.int64),
            5,
            1,
            [4, 8],
            offset_capture_sizes=[1],
            tensor_parallel_size=2,
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, utils.NO_SPLIT_TP_SHAPE_UNSUPPORTED)

    def test_padding_sentinel_contract(self):
        self.assertEqual(utils.padding_fill_value("slot_mapping"), -1)
        self.assertEqual(utils.padding_fill_value("block_table_tensor"), 0)
        self.assertEqual(utils.padding_fill_value("positions"), 0)

    def test_metadata_replace_preserves_unsliced_fields(self):
        @dataclass
        class Metadata:
            seq_lens: tuple[int, ...]
            kvcomp_metadata: object
            num_logits_indices: int

        marker = object()
        original = Metadata((1, 2), marker, 17)
        split = utils.replace_split_metadata(original, seq_lens=(2,))
        self.assertEqual(split.seq_lens, (2,))
        self.assertIs(split.kvcomp_metadata, marker)
        self.assertEqual(split.num_logits_indices, 17)

    def test_disabled_plan_does_not_require_stabilization(self):
        self.assertFalse(utils.should_stabilize_inplace_metadata(None))
        self.assertTrue(utils.should_stabilize_inplace_metadata(object()))

    def test_runtime_graph_integration_uses_final_core_schema_only(self):
        root = Path(__file__).parents[3]
        runner_source = (root / "vllm_ascend/worker/inplace_split_runner.py").read_text()
        context_source = (root / "vllm_ascend/ascend_forward_context.py").read_text()

        for required in (
            "CUDAGraphRuntimeMetadata(",
            "runtime_metadata=runtime_metadata",
            "allow_runtime_key_registration=",
            "is_secondary_stream=",
            "allow_runtime_graph_capture=",
        ):
            self.assertIn(required, runner_source + context_source)
        for removed in (
            "allow_inplace_lazy_key=",
            "graph_variant=(",
            "attention_backend=(split_attention_backend",
            "capture_metadata_mode=capture_metadata_mode",
        ):
            self.assertNotIn(removed, runner_source)


if __name__ == "__main__":
    unittest.main()
