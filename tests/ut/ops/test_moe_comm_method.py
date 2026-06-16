from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    MC2CommImpl,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEFusedExpertsInput,
    MoEOffloadParams,
    MoEPrepareOutput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput
from vllm_ascend.moe_offload.runtime import MoeOffloadDecisionPath
from vllm_ascend.quantization.methods.base import QuantType


class TestMoECommMethod(TestBase):
    def setUp(self):
        self._fusion_patcher = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.set_gmmswigluquant_method",
            return_value=False,
        )
        self._fusion_patcher.start()
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.global_redundant_expert_num = 0

    def tearDown(self):
        self._fusion_patcher.stop()

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_all_gather_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2")
    def test_mc2_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "mc2"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=torch.tensor([1, 0, 1, 0]),
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = MC2CommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AlltoAllCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        _ = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp")
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_method(
        self, mock_unified_apply_mlp, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        dispatch_topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(6, 8),
            group_list=torch.tensor([2, 2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=dispatch_topk_weights,
                expanded_row_idx=torch.arange(8, dtype=torch.int32),
                restore_shape=torch.Size([4, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(4, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        # Mock unified_apply_mlp
        mock_unified_apply_mlp.return_value = torch.randn(6, 8)

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test fused_experts method
        hidden_states = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        topk_weights = dispatch_topk_weights
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])

        # Make sure tensors are contiguous and have correct strides
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        result = comm_impl.fused_experts(
            fused_experts_input=MoEFusedExpertsInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                weights=MoEWeights(
                    w1=[w1],
                    w2=[w2],
                ),
                routing=MoERoutingParams(
                    expert_map=None,
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                activation="silu",
                need_trans=False,
                dynamic_eplb=False,
                quant=MoEQuantParams(),
            )
        )

        # Verify result shape
        self.assertEqual(result.routed_out.shape, (4, 8))

        # Verify token_dispatch was called
        mock_td_instance.token_dispatch.assert_called_once()

        # Verify unified_apply_mlp was called
        mock_unified_apply_mlp.assert_called_once()
        mlp_compute_input = mock_unified_apply_mlp.call_args.kwargs["mlp_compute_input"]
        self.assertFalse(mlp_compute_input.fusion)
        self.assertFalse(mlp_compute_input.quant.is_mxfp)

        # Verify token_combine was called
        mock_td_instance.token_combine.assert_called_once_with(
            hidden_states=mock_unified_apply_mlp.return_value,
            combine_metadata=mock_td_instance.token_dispatch.return_value.combine_metadata,
        )

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.set_gmmswigluquant_method", return_value=False)
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_applies_moe_offload_slot_plan_at_boundary(
        self, mock_set_fusion, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        del mock_set_fusion
        del mock_prepare_finalize
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        dispatch_topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        mock_td_instance = MagicMock()
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(4, 8),
            group_list=torch.tensor([2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=dispatch_topk_weights,
                expanded_row_idx=torch.arange(4, dtype=torch.int32),
                restore_shape=torch.Size([2, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(2, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        comm_impl = AllGatherCommImpl(self.moe_config)
        topk_ids = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
        original_w1 = torch.randn(3, 2, 4)
        original_w2 = torch.randn(3, 4, 2)
        slot_w1 = torch.randn(2, 2, 4)
        slot_w2 = torch.randn(2, 4, 2)
        slot_log2phy = torch.tensor([-1, 0, 1], dtype=torch.int32)
        prepared = MagicMock(
            w1=slot_w1,
            w2=slot_w2,
            log2phy=slot_log2phy,
            physical_expert_count=2,
        )
        runtime = MagicMock()
        runtime.should_use_layered_runtime = False
        runtime.config.graph_compatible_offload = False
        runtime.config.phase_split_enabled = False
        runtime.prepare_fixed_slot_plan.return_value = prepared
        runtime.should_use_fixed_slot_plan_for_layer.return_value = True

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_offload_runtime", return_value=runtime),
            patch.object(comm_impl, "_apply_mlp", return_value=torch.randn(4, 8)) as mock_apply_mlp,
        ):
            comm_impl.fused_experts(
                fused_experts_input=MoEFusedExpertsInput(
                    hidden_states=torch.randn(2, 8),
                    topk_weights=dispatch_topk_weights,
                    topk_ids=topk_ids,
                    weights=MoEWeights(w1=original_w1, w2=original_w2),
                    routing=MoERoutingParams(
                        expert_map=None,
                        global_redundant_expert_num=0,
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    activation="silu",
                    need_trans=False,
                    dynamic_eplb=False,
                    quant=MoEQuantParams(),
                    offload=MoEOffloadParams(
                        enabled=True,
                        layer_id=6,
                        num_logical_experts=3,
                        expected_device_type="cpu",
                    ),
                )
            )

        runtime.prepare_fixed_slot_plan.assert_called_once_with(
            layer_id=6,
            active_experts=(1, 2),
            num_logical_experts=3,
            device=topk_ids.device,
        )
        prepared.validate_backend_ready.assert_called_once_with(expected_device_type="cpu")
        dispatch_input = mock_td_instance.token_dispatch.call_args.kwargs["token_dispatch_input"]
        self.assertTrue(torch.equal(dispatch_input.topk_ids, slot_log2phy[topk_ids]))
        mlp_compute_input = mock_apply_mlp.call_args.args[0]
        self.assertIs(mlp_compute_input.weights.w1, slot_w1)
        self.assertIs(mlp_compute_input.weights.w2, slot_w2)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.set_gmmswigluquant_method", return_value=False)
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_keeps_full_weights_for_high_fanout_offload_decision(
        self, mock_set_fusion, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        del mock_set_fusion, mock_prepare_finalize
        mock_get_forward_context.return_value = MagicMock(moe_comm_method="all_gather")
        mock_td_instance = MagicMock()
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(4, 8),
            group_list=torch.tensor([2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=torch.ones(2, 2),
                expanded_row_idx=torch.arange(4, dtype=torch.int32),
                restore_shape=torch.Size([2, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(2, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        comm_impl = AllGatherCommImpl(self.moe_config)
        original_w1 = torch.randn(3, 2, 4)
        original_w2 = torch.randn(3, 4, 2)
        runtime = MagicMock()
        runtime.should_use_layered_runtime = True
        runtime.config.graph_compatible_offload = False
        runtime.config.phase_split_enabled = False
        runtime.decide_layered_path.return_value = MagicMock(path=MoeOffloadDecisionPath.FULL_WEIGHT_PATH)

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_offload_runtime", return_value=runtime),
            patch.object(comm_impl, "_apply_mlp", return_value=torch.randn(4, 8)) as mock_apply_mlp,
        ):
            comm_impl.fused_experts(
                fused_experts_input=MoEFusedExpertsInput(
                    hidden_states=torch.randn(2, 8),
                    topk_weights=torch.ones(2, 2),
                    topk_ids=torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
                    weights=MoEWeights(w1=original_w1, w2=original_w2),
                    routing=MoERoutingParams(
                        expert_map=None,
                        global_redundant_expert_num=0,
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(),
                    offload=MoEOffloadParams(
                        enabled=True,
                        layer_id=6,
                        num_logical_experts=3,
                        expected_device_type="cpu",
                    ),
                )
            )

        runtime.prepare_fixed_slot_plan.assert_not_called()
        mlp_compute_input = mock_apply_mlp.call_args.args[0]
        self.assertIs(mlp_compute_input.weights.w1, original_w1)
        self.assertIs(mlp_compute_input.weights.w2, original_w2)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.set_gmmswigluquant_method", return_value=False)
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_fails_closed_at_boundary_for_unavailable_full_weights(
        self, mock_set_fusion, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        del mock_set_fusion, mock_prepare_finalize
        mock_get_forward_context.return_value = MagicMock(moe_comm_method="all_gather")
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance
        comm_impl = AllGatherCommImpl(self.moe_config)
        runtime = MagicMock()
        runtime.should_use_layered_runtime = True
        runtime.config.graph_compatible_offload = False
        runtime.config.phase_split_enabled = False
        runtime.decide_layered_path.return_value = MagicMock(
            path=MoeOffloadDecisionPath.FAIL_CLOSED,
            reason="high_fanout_full_weights_unavailable",
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_offload_runtime", return_value=runtime),
            self.assertRaisesRegex(RuntimeError, "high_fanout_full_weights_unavailable"),
        ):
            comm_impl.fused_experts(
                fused_experts_input=MoEFusedExpertsInput(
                    hidden_states=torch.randn(2, 8),
                    topk_weights=torch.ones(2, 2),
                    topk_ids=torch.tensor([[0, 1], [2, 1]], dtype=torch.int32),
                    weights=MoEWeights(w1=torch.randn(3, 2, 4), w2=torch.randn(3, 4, 2)),
                    routing=MoERoutingParams(
                        expert_map=None,
                        global_redundant_expert_num=0,
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(),
                    offload=MoEOffloadParams(
                        enabled=True,
                        layer_id=6,
                        num_logical_experts=3,
                        expected_device_type="cpu",
                    ),
                )
            )

        mock_td_instance.token_dispatch.assert_not_called()
