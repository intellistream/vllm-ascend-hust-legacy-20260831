from unittest import mock

from vllm.config import VllmConfig

from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


def _config(*, architecture: str, quantization: str | None) -> mock.MagicMock:
    config = mock.MagicMock(spec=VllmConfig)
    config.model_config = mock.MagicMock()
    config.model_config.architectures = [architecture]
    config.model_config.quantization = quantization
    config.compilation_config = mock.MagicMock()
    config.compilation_config.pass_config = mock.MagicMock(enable_sp=False)
    config.additional_config = {
        "ascend_compilation_config": {
            "fuse_norm_quant": True,
            "fuse_qknorm_rope": False,
            "fuse_allreduce_rms": False,
            "fuse_muls_add": False,
        }
    }
    return config


@mock.patch("vllm_ascend.utils.is_310p", return_value=False)
@mock.patch("vllm_ascend.compilation.passes.norm_quant_fusion_pass.AddRMSNormQuantFusionPass")
def test_skips_norm_quant_fusion_for_ascend_quantized_deepseek_v2(
    norm_quant_pass: mock.MagicMock,
    _is_310p: mock.MagicMock,
) -> None:
    manager = GraphFusionPassManager()

    with mock.patch("vllm_ascend.compilation.graph_fusion_pass_manager.logger.warning") as warning:
        manager.configure(
            _config(
                architecture="DeepseekV2ForCausalLM",
                quantization="ascend",
            )
        )

    norm_quant_pass.assert_not_called()
    warning.assert_called_once()


@mock.patch("vllm_ascend.utils.is_310p", return_value=False)
@mock.patch("vllm_ascend.compilation.passes.norm_quant_fusion_pass.AddRMSNormQuantFusionPass")
def test_keeps_norm_quant_fusion_for_other_models(
    norm_quant_pass: mock.MagicMock,
    _is_310p: mock.MagicMock,
) -> None:
    config = _config(architecture="Qwen3ForCausalLM", quantization="ascend")
    manager = GraphFusionPassManager()

    manager.configure(config)

    norm_quant_pass.assert_called_once_with(config)
