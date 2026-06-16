import unittest
from typing import ClassVar
from unittest.mock import patch

import torch

from vllm_ascend.moe_offload.compute_bucket import (
    ComputeBucket,
    ComputeBucketDecision,
    ComputeBucketDecisionPath,
)
from vllm_ascend.ops.fused_moe.moe_mlp import (
    cumsum_group_list,
    ComputeBucketFastPathGate,
    clear_active_expert_compaction_cache,
    maybe_compact_active_expert_unquant_path,
    maybe_select_compute_bucket_fast_path,
    unified_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_request_unquant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=torch.randn(1, 16),
                w2_bias=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        mock_unquant.assert_called_once()
        self.assertEqual(mock_unquant.call_args.kwargs["activation"], "silu")
        self.assertFalse(mock_unquant.call_args.kwargs["need_trans"])
        mock_quant.assert_not_called()

    def test_request_quant_path(self):
        for quant_type, mxfp_dtype in (
            (QuantType.MXFP8, torch.float8_e4m3fn),
            (QuantType.MXFP4, MXFP4_TEST_DTYPE),
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(2, 8)
                expected = torch.randn(2, 8)
                mlp_compute_input = MoEMlpComputeInput(
                    hidden_states=hidden_states,
                    group_list=torch.tensor([2, 2], dtype=torch.int64),
                    group_list_type=1,
                    dynamic_scale=torch.randn(2, 1),
                    topk_scales=None,
                    weights=MoEWeights(
                        w1=torch.randn(1, 16, 8),
                        w2=torch.randn(1, 8, 8),
                        w1_scale=[torch.randn(1)],
                        w2_scale=[torch.randn(1)],
                    ),
                    quant=MoEQuantParams(
                        quant_type=quant_type,
                        mxfp=MoEMxfpParams(
                            act_quant_type=mxfp_dtype,
                            weight_quant_type=mxfp_dtype,
                            use_bf16=False,
                        ),
                    ),
                    fusion=True,
                    activation="silu",
                    need_trans=False,
                    dynamic_eplb=True,
                )

                with (
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
                ):
                    output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

                self.assertTrue(output is expected)
                mock_quant.assert_called_once()
                quant_kwargs = mock_quant.call_args.kwargs
                self.assertTrue(quant_kwargs["use_mxfp_quant"])
                self.assertTrue(quant_kwargs["fusion"])
                self.assertTrue(quant_kwargs["dynamic_eplb"])
                self.assertEqual(quant_kwargs["act_quant_type"], mxfp_dtype)
                self.assertEqual(quant_kwargs["weight_quant_type"], mxfp_dtype)
                self.assertFalse(quant_kwargs["use_bf16"])
                mock_unquant.assert_not_called()

    def test_request_records_compute_bucket_fast_path_gate_before_unquant_fallback(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="counts:2,0,0",
            reason="signature_matched",
            phase="decode",
            bucket_id=7,
        )
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 0, 0], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(3, 16, 8),
                w2=torch.randn(3, 8, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
            compute_bucket_decision=decision,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.record_compute_bucket_fast_path_gate") as mock_record_gate,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        mock_unquant.assert_called_once()
        mock_record_gate.assert_called_once()
        gate = mock_record_gate.call_args.args[0]
        self.assertTrue(gate.enabled)
        self.assertEqual(gate.bucket_id, 7)
        self.assertEqual(gate.signature, "counts:2,0,0")
        self.assertEqual(gate.original_expert_count, 3)
        self.assertEqual(gate.compact_expert_count, 1)

    def test_request_compacts_active_expert_weights_for_unquant_bucket(self):
        hidden_states = torch.randn(3, 8)
        expected = torch.randn(3, 8)
        w1 = torch.randn(4, 16, 8)
        w2 = torch.randn(4, 8, 8)
        w1_bias = torch.randn(4, 16)
        w2_bias = torch.randn(4, 8)
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="counts:2,0,1,0",
            reason="signature_matched",
            phase="decode",
            bucket_id=5,
            bucket=ComputeBucket(
                bucket_id=5,
                signature="counts:2,0,1,0",
                active_expert_ids=(0, 2),
                compact_group_list=(2, 1),
                original_expert_count=4,
            ),
        )
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=w1,
                w2=w2,
                w1_bias=w1_bias,
                w2_bias=w2_bias,
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
            compute_bucket_decision=decision,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.record_compute_bucket_fast_path_gate"),
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        kwargs = mock_unquant.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["group_list"], torch.tensor([2, 1], dtype=torch.int64)))
        self.assertEqual(kwargs["group_list_type"], 1)
        self.assertTrue(torch.equal(kwargs["w1"], w1[[0, 2]]))
        self.assertTrue(torch.equal(kwargs["w2"], w2[[0, 2]]))
        self.assertTrue(torch.equal(kwargs["w1_bias"], w1_bias[[0, 2]]))
        self.assertTrue(torch.equal(kwargs["w2_bias"], w2_bias[[0, 2]]))


class TestComputeBucketFastPathGate(unittest.TestCase):
    def _mlp_input(
        self,
        *,
        decision: ComputeBucketDecision | None,
        quant_type: QuantType = QuantType.NONE,
        group_list_type: int = 1,
    ) -> MoEMlpComputeInput:
        return MoEMlpComputeInput(
            hidden_states=torch.randn(1, 8),
            group_list=torch.tensor([1, 0, 1], dtype=torch.int64),
            group_list_type=group_list_type,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(3, 16, 8),
                w2=torch.randn(3, 8, 8),
            ),
            quant=MoEQuantParams(quant_type=quant_type),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
            compute_bucket_decision=decision,
        )

    def test_gate_selects_planned_unquantized_bucket(self):
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="counts:1,0,1",
            reason="signature_matched",
            phase="decode",
            bucket_id=2,
        )

        gate = maybe_select_compute_bucket_fast_path(self._mlp_input(decision=decision))

        self.assertTrue(gate.enabled)
        self.assertEqual(gate.bucket_id, 2)
        self.assertEqual(gate.reason, "eligible")
        self.assertEqual(gate.original_expert_count, 3)
        self.assertEqual(gate.compact_expert_count, 2)

    def test_gate_falls_back_without_bucket_or_for_quantized_path(self):
        fallback_gate = maybe_select_compute_bucket_fast_path(self._mlp_input(decision=None))
        quant_gate = maybe_select_compute_bucket_fast_path(
            self._mlp_input(
                decision=ComputeBucketDecision(
                    path=ComputeBucketDecisionPath.BUCKET,
                    signature="counts:1,0,1",
                    reason="signature_matched",
                    phase="decode",
                    bucket_id=2,
                ),
                quant_type=QuantType.W8A8,
            ))

        self.assertFalse(fallback_gate.enabled)
        self.assertEqual(fallback_gate.reason, "no_bucket_decision")
        self.assertFalse(quant_gate.enabled)
        self.assertEqual(quant_gate.reason, "requires_unquantized_path")


class TestActiveExpertCompaction(unittest.TestCase):
    def setUp(self):
        clear_active_expert_compaction_cache()

    def test_compaction_uses_bucket_active_expert_plan(self):
        gate = ComputeBucketFastPathGate(
            enabled=True,
            reason="eligible",
            bucket_id=5,
            signature="counts:2,0,1,0",
        )
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="counts:2,0,1,0",
            reason="signature_matched",
            phase="decode",
            bucket_id=5,
            bucket=ComputeBucket(
                bucket_id=5,
                signature="counts:2,0,1,0",
                active_expert_ids=(0, 2),
                compact_group_list=(2, 1),
                original_expert_count=4,
            ),
        )
        w1 = torch.randn(4, 16, 8)
        w2 = torch.randn(4, 8, 8)

        compaction = maybe_compact_active_expert_unquant_path(
            gate=gate,
            decision=decision,
            w1=w1,
            w2=w2,
            group_list=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
            group_list_type=1,
            w1_bias=None,
            w2_bias=None,
        )

        self.assertIsNotNone(compaction)
        self.assertTrue(torch.equal(compaction.group_list, torch.tensor([2, 1], dtype=torch.int64)))
        self.assertTrue(torch.equal(compaction.w1, w1[[0, 2]]))

    def test_compaction_skips_when_all_experts_are_active(self):
        gate = ComputeBucketFastPathGate(enabled=True, reason="eligible")
        w1 = torch.randn(2, 16, 8)
        w2 = torch.randn(2, 8, 8)

        compaction = maybe_compact_active_expert_unquant_path(
            gate=gate,
            decision=None,
            w1=w1,
            w2=w2,
            group_list=torch.tensor([1, 2], dtype=torch.int64),
            group_list_type=1,
            w1_bias=None,
            w2_bias=None,
        )

        self.assertIsNone(compaction)

    def test_compaction_skips_for_cumsum_group_list_until_supported(self):
        gate = ComputeBucketFastPathGate(enabled=True, reason="eligible")
        w1 = torch.randn(4, 16, 8)
        w2 = torch.randn(4, 8, 8)

        compaction = maybe_compact_active_expert_unquant_path(
            gate=gate,
            decision=None,
            w1=w1,
            w2=w2,
            group_list=torch.tensor([2, 2, 3, 3], dtype=torch.int64),
            group_list_type=0,
            w1_bias=None,
            w2_bias=None,
        )

        self.assertIsNone(compaction)

    def test_compaction_uses_planned_cumsum_bucket(self):
        gate = ComputeBucketFastPathGate(
            enabled=True,
            reason="eligible",
            bucket_id=6,
            signature="cumsum:0,2,2,5",
        )
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="cumsum:0,2,2,5",
            reason="signature_matched",
            phase="decode",
            bucket_id=6,
            bucket=ComputeBucket(
                bucket_id=6,
                signature="cumsum:0,2,2,5",
                active_expert_ids=(1, 3),
                compact_group_list=(2, 5),
                original_expert_count=4,
            ),
        )
        w1 = torch.randn(4, 16, 8)
        w2 = torch.randn(4, 8, 8)
        w1_bias = torch.randn(4, 16)
        w2_bias = torch.randn(4, 8)

        compaction = maybe_compact_active_expert_unquant_path(
            gate=gate,
            decision=decision,
            w1=w1,
            w2=w2,
            group_list=torch.tensor([0, 2, 2, 5], dtype=torch.int64),
            group_list_type=0,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
        )

        self.assertIsNotNone(compaction)
        self.assertTrue(torch.equal(compaction.group_list, torch.tensor([2, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(compaction.w1, w1[[1, 3]]))
        self.assertTrue(torch.equal(compaction.w2, w2[[1, 3]]))
        self.assertTrue(torch.equal(compaction.w1_bias, w1_bias[[1, 3]]))
        self.assertTrue(torch.equal(compaction.w2_bias, w2_bias[[1, 3]]))

    def test_compaction_reuses_planned_device_tensors_for_same_bucket(self):
        gate = ComputeBucketFastPathGate(
            enabled=True,
            reason="eligible",
            bucket_id=5,
            signature="counts:2,0,1,0",
        )
        decision = ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature="counts:2,0,1,0",
            reason="signature_matched",
            phase="decode",
            bucket_id=5,
            bucket=ComputeBucket(
                bucket_id=5,
                signature="counts:2,0,1,0",
                active_expert_ids=(0, 2),
                compact_group_list=(2, 1),
                original_expert_count=4,
            ),
        )
        w1 = torch.randn(4, 16, 8)
        w2 = torch.randn(4, 8, 8)
        group_list = torch.tensor([2, 0, 1, 0], dtype=torch.int64)

        with patch("vllm_ascend.ops.fused_moe.moe_mlp.torch.tensor", wraps=torch.tensor) as mock_tensor:
            first = maybe_compact_active_expert_unquant_path(
                gate=gate,
                decision=decision,
                w1=w1,
                w2=w2,
                group_list=group_list,
                group_list_type=1,
                w1_bias=None,
                w2_bias=None,
            )
            second = maybe_compact_active_expert_unquant_path(
                gate=gate,
                decision=decision,
                w1=w1,
                w2=w2,
                group_list=group_list,
                group_list_type=1,
                w1_bias=None,
                w2_bias=None,
            )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(mock_tensor.call_count, 2)
        self.assertTrue(torch.equal(first.group_list, second.group_list))


if __name__ == "__main__":
    unittest.main(verbosity=2)
