# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.base import AscendMoEScheme, QuantType
from vllm_ascend.quantization.methods.registry import get_scheme_class
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.methods.w8a8_online import (
    AscendW8A8OnlineFusedMoEMethod,
)


class TestAscendW8A8OnlineFusedMoEMethod(TestBase):
    num_experts = 4
    hidden_size = 32
    intermediate_size = 64

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def setUp(self, mock_get_ascend_config, mock_get_mc2_group, mock_get_rank):
        with patch(
            "vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config"
        ) as mock_get_current_vllm_config:
            mock_vllm_config = Mock()
            mock_vllm_config.model_config.dtype = torch.float16
            mock_vllm_config.model_config.enforce_eager = True
            mock_vllm_config.compilation_config.mode = Mock()
            mock_get_current_vllm_config.return_value = mock_vllm_config
            mock_ascend_config = Mock()
            mock_ascend_config.multistream_overlap_gate = False
            mock_ascend_config.eplb_config = Mock(dynamic_eplb=False)
            mock_ascend_config.enable_fused_mc2 = 0
            mock_get_ascend_config.return_value = mock_ascend_config
            mock_get_mc2_group.return_value = Mock(device_group=Mock())
            mock_get_rank.return_value = 0
            self.quant_method = AscendW8A8OnlineFusedMoEMethod()

    def test_is_registered_as_w8a8_online_moe(self):
        cls = get_scheme_class("W8A8_ONLINE", "moe")
        self.assertIs(cls, AscendW8A8OnlineFusedMoEMethod)

    def test_subclasses_dynamic_moe(self):
        self.assertTrue(issubclass(AscendW8A8OnlineFusedMoEMethod, AscendW8A8DynamicFusedMoEMethod))
        self.assertTrue(issubclass(AscendW8A8OnlineFusedMoEMethod, AscendMoEScheme))
        self.assertEqual(self.quant_method.quant_type, QuantType.W8A8)

    def test_get_weight_loads_floating_point(self):
        # Online int8 loads fp16/bf16 weights (not int8) so they can be
        # quantized during process_weights_after_loading.
        for dtype in (torch.float16, torch.bfloat16):
            param_dict = self.quant_method.get_weight(self.num_experts, self.intermediate_size, self.hidden_size, dtype)
            self.assertEqual(param_dict["w13_weight"].dtype, dtype)
            self.assertEqual(param_dict["w2_weight"].dtype, dtype)
            self.assertEqual(
                param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
            )
            self.assertEqual(
                param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size)
            )

    def test_quantize_per_row_produces_int8_and_scale(self):
        torch.manual_seed(0)
        w = torch.randn(2, 5, 8, dtype=torch.float32)
        q, scale = AscendW8A8OnlineFusedMoEMethod._quantize_per_row(w)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(q.shape, w.shape)
        self.assertEqual(scale.shape, (2, 5, 1))
        self.assertTrue(q.abs().max().item() <= 127)
        # Dequantization error must be small for per-row symmetric quant.
        deq = q.to(torch.float32) * scale
        self.assertLess((deq - w).abs().max().item(), 0.5)

    def test_quantize_per_row_zero_row_no_nan(self):
        w = torch.zeros(1, 3, 4, dtype=torch.float32)
        q, _ = AscendW8A8OnlineFusedMoEMethod._quantize_per_row(w)
        self.assertFalse(torch.isnan(q.to(torch.float32)).any())

    def _build_fp16_layer(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.randn(self.num_experts, 2 * self.intermediate_size, self.hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size, self.intermediate_size, dtype=torch.float16),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, 1, dtype=torch.float16), requires_grad=False
        )
        layer.w13_weight_offset = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, 1, dtype=torch.float16), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 1, dtype=torch.float16), requires_grad=False
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 1, dtype=torch.float16), requires_grad=False
        )
        return layer

    @patch("torch_npu.npu_format_cast")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def test_process_weights_quantizes_fp16_to_int8(self, mock_get_ascend_config, mock_npu_format_cast):
        mock_get_ascend_config.return_value = Mock(enable_fused_mc2=0)
        mock_npu_format_cast.side_effect = lambda weight, _: weight

        layer = self._build_fp16_layer()
        w13_fp16 = layer.w13_weight.data.clone()
        self.quant_method.process_weights_after_loading(layer)

        # Weights must be quantized to int8 after loading.
        self.assertEqual(layer.w13_weight.data.dtype, torch.int8)
        self.assertEqual(layer.w2_weight.data.dtype, torch.int8)
        # Per-channel scales must be populated and flattened.
        self.assertEqual(layer.w13_weight_scale_fp32.shape, (self.num_experts, 2 * self.intermediate_size))
        # Symmetric quantization -> offset is zero.
        self.assertTrue(torch.equal(layer.w13_weight_offset.data, torch.zeros_like(layer.w13_weight_offset.data)))
        # NZ format cast must have been invoked for W8A8.
        mock_npu_format_cast.assert_called()
        # Sanity: dequantized weights approximate the original fp16 weights.
        deq = layer.w13_weight.data.to(torch.float32) * layer.w13_weight_scale_fp32.data.unsqueeze(1).to(torch.float32)
        max_err = (deq - w13_fp16.transpose(1, 2).to(torch.float32)).abs().max().item()
        self.assertLess(max_err, 0.5)

    @patch("torch_npu.npu_format_cast")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def test_process_weights_after_loading_is_idempotent(self, mock_get_ascend_config, mock_npu_format_cast):
        mock_get_ascend_config.return_value = Mock(enable_fused_mc2=0)
        mock_npu_format_cast.side_effect = lambda weight, _: weight

        layer = self._build_fp16_layer()
        self.quant_method.process_weights_after_loading(layer)
        w13_first = layer.w13_weight.data.clone()
        w13_scale_first = layer.w13_weight_scale_fp32.data.clone()
        nz_cast_count = mock_npu_format_cast.call_count

        # A second call must be a no-op: no re-quantization, no re-transpose,
        # no extra NZ cast (mirrors upstream _already_called_... guard).
        self.quant_method.process_weights_after_loading(layer)

        self.assertTrue(getattr(layer, "_already_called_process_weights_after_loading", False))
        self.assertTrue(torch.equal(layer.w13_weight.data, w13_first))
        self.assertTrue(torch.equal(layer.w13_weight_scale_fp32.data, w13_scale_first))
        self.assertEqual(mock_npu_format_cast.call_count, nz_cast_count)
