#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from torch import fx as fx
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.logger import logger


class GraphFusionPassManager:
    """
    A pass manager for graph fusion passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu
    does not support triton for now, we define our own pass manager.
    """

    def __init__(self):
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph) -> fx.Graph:
        compile_range = get_pass_context().compile_range

        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    def _add_optional_pass(self, pass_name: str, pass_factory):
        try:
            self.passes.append(pass_factory())
        except RuntimeError as exc:
            message = str(exc)
            if "not in libopapi.so" not in message and "libopapi.so" not in message:
                raise
            logger.warning(
                "Skipping Ascend graph fusion pass %s because the required "
                "OPAPI symbol is unavailable in this runtime: %s",
                pass_name,
                exc,
            )

    def configure(self, config: VllmConfig):
        from vllm_ascend.utils import is_310p

        # By default, we enable the graph fusion and quantization fusion pass.
        self.ascend_compilation_config: dict = config.additional_config.get("ascend_compilation_config", {})
        if self.ascend_compilation_config.get("fuse_norm_quant", True) and not is_310p():
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self._add_optional_pass("fuse_norm_quant", lambda: AddRMSNormQuantFusionPass(config))

        if self.ascend_compilation_config.get("fuse_qknorm_rope", True):
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))

        if self.ascend_compilation_config.get("fuse_allreduce_rms", True):
            from .passes.allreduce_rmsnorm_fusion_pass import MatmulAllReduceAddRMSNormPass

            self._add_optional_pass("fuse_allreduce_rms", lambda: MatmulAllReduceAddRMSNormPass(config))

        if self.ascend_compilation_config.get("fuse_muls_add", True) and not is_310p():
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        if config.compilation_config.pass_config.enable_sp:
            from .passes.sequence_parallelism import SequenceParallelismPass
            from .passes.sequence_parallelism_moe import SequenceParallelismMoePass

            self._add_optional_pass("sequence_parallelism", lambda: SequenceParallelismPass(config))
            self._add_optional_pass("sequence_parallelism_moe", lambda: SequenceParallelismMoePass(config))
