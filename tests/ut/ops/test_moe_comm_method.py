from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    FusedMC2CommImpl,
    MC2CommImpl,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEFusedExpertsInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput, TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.utils import AscendDeviceType


class TestMoECommMethod(TestBase):
    def setUp(self):
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant = False
        self.mock_ascend_config.enable_fused_mc2 = False
        self._patch_get_ascend_config = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_module = patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self._patch_get_ascend_config_module.start()
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.pcp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.global_redundant_expert_num = 0

    def tearDown(self):
        self._patch_get_ascend_config.stop()
        self._patch_get_ascend_config_module.stop()

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

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_tokens_capacity", return_value=8192)
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_device_type", return_value=AscendDeviceType.A2)
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_tensor_model_parallel_world_size", return_value=2)
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_ep_group", return_value=MagicMock(world_size=2))
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2")
    @patch("torch.zeros", return_value=MagicMock())
    def test_a2_fused_mc2_reserves_scheduler_routed_domain(
        self,
        mock_zeros,
        mock_token_dispatcher,
        mock_prepare_finalize,
        mock_ep_group,
        mock_tp_size,
        mock_device_type,
        mock_capacity,
    ):
        self.mock_ascend_config.enable_fused_mc2 = 1
        self.moe_config.tp_size = 2
        self.moe_config.ep_size = 2
        self.moe_config.experts_per_token = 8

        comm_impl = FusedMC2CommImpl(self.moe_config)

        self.assertEqual(comm_impl.max_output_size, 65536)

    @patch("torch.ops._C_ascend.dispatch_ffn_combine", create=True)
    def test_a2_fused_mc2_passes_reserved_routed_domain_to_op(self, mock_dispatch_ffn_combine):
        self.mock_ascend_config.enable_fused_mc2 = 1
        comm_impl = object.__new__(FusedMC2CommImpl)
        comm_impl.max_output_size = 64
        comm_impl.expert_token_nums = torch.zeros(2, dtype=torch.int32)
        comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
        comm_impl.token_dispatcher.moe_all_to_all_group_name = "test-group"

        fused_input = MagicMock()
        fused_input.hidden_states = torch.randn(4, 8)
        fused_input.topk_ids = torch.zeros((4, 2), dtype=torch.int32)
        fused_input.topk_weights = torch.full((4, 2), 0.5)
        fused_input.routing.log2phy = None
        fused_input.routing.mc2_mask = torch.ones(4, dtype=torch.bool)
        fused_input.weights.w1 = [torch.empty(0)]
        fused_input.weights.w2 = [torch.empty(0)]
        fused_input.weights.w1_scale = [torch.empty(0)]
        fused_input.weights.w2_scale = [torch.empty(0)]
        fused_input.weights.w1_scale_bias = [torch.empty(0)]
        fused_input.weights.w2_scale_bias = [torch.empty(0)]
        fused_input.swiglu_limit = 1_000_000.0

        comm_impl.fused_experts(fused_input)

        self.assertEqual(mock_dispatch_ffn_combine.call_args.kwargs["max_output_size"], 64)

        fused_input.hidden_states = torch.randn(33, 8)
        fused_input.topk_ids = torch.zeros((33, 2), dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "workspace profiled at startup"):
            comm_impl.fused_experts(fused_input)

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

        # Mock unified_apply_mlp returns (tensor, event) tuple
        mock_unified_apply_mlp.return_value = (torch.randn(6, 8), MagicMock())

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
            hidden_states=mock_unified_apply_mlp.return_value[0],
            combine_metadata=mock_td_instance.token_dispatch.return_value.combine_metadata,
        )
