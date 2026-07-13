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

import json
import time
from pathlib import Path

import torch
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
        self.ascend_compilation_config: dict = {}

    def _mlp_classifier_enabled(self) -> bool:
        value = self.ascend_compilation_config.get("mlp_materialization_classify")
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _emit_mlp_classifier_event(self, payload: dict) -> None:
        if not self._mlp_classifier_enabled():
            return
        output_file = self.ascend_compilation_config.get("mlp_materialization_classify_file")
        if not output_file:
            return
        event = {
            "timestamp_ns": time.time_ns(),
            "evidence_label": "real-compile-pass-probe",
            "pass": self.__class__.__name__,
            **payload,
        }
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def __call__(self, graph: fx.Graph) -> fx.Graph:
        compile_range = get_pass_context().compile_range
        fx_graph = graph.graph if hasattr(graph, "graph") else graph
        self._emit_mlp_classifier_event(
            {
                "event": "graph_fusion_call",
                "manager_instance_id": id(self),
                "graph_type": graph.__class__.__name__,
                "node_count": sum(1 for _ in fx_graph.nodes),
                "compile_range": {
                    "start": getattr(compile_range, "start", None),
                    "end": getattr(compile_range, "end", None),
                },
                "passes": [pass_.__class__.__name__ for pass_ in self.passes],
            }
        )

        for pass_ in self.passes:
            pass_name = pass_.__class__.__name__
            is_applicable = pass_.is_applicable_for_range(compile_range)
            self._emit_mlp_classifier_event(
                {
                    "event": "graph_fusion_pass_applicability",
                    "manager_instance_id": id(self),
                    "pass_name": pass_name,
                    "is_applicable": is_applicable,
                    "compile_range": {
                        "start": getattr(compile_range, "start", None),
                        "end": getattr(compile_range, "end", None),
                    },
                }
            )
            if is_applicable:
                pass_(graph)
        graph.recompile()
        self._emit_mlp_classifier_event(
            {
                "event": "graph_fusion_done",
                "manager_instance_id": id(self),
                "node_count": sum(1 for _ in fx_graph.nodes),
            }
        )
        return graph

    def add(self, pass_: VllmInductorPass):
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    def configure(self, config: VllmConfig):
        from vllm_ascend.utils import is_310p

        # By default, we enable the graph fusion and quantization fusion pass.
        self.ascend_compilation_config: dict = config.additional_config.get("ascend_compilation_config", {})
        if self.ascend_compilation_config.get("fuse_norm_quant", True) and not is_310p():
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self.passes.append(AddRMSNormQuantFusionPass(config))

        if self.ascend_compilation_config.get("fuse_qknorm_rope", True) and hasattr(
            torch.ops.vllm, "qkv_rmsnorm_rope"
        ):
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))
        elif self.ascend_compilation_config.get("fuse_qknorm_rope", True):
            logger.warning(
                "Skipping qknorm_rope fusion because torch.ops.vllm.qkv_rmsnorm_rope "
                "is not registered in this runtime."
            )

        if self.ascend_compilation_config.get("fuse_allreduce_rms", True):
            from .passes.allreduce_rmsnorm_fusion_pass import MatmulAllReduceAddRMSNormPass

            self.passes.append(MatmulAllReduceAddRMSNormPass(config))

        if self.ascend_compilation_config.get("fuse_muls_add", True) and not is_310p():
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        from .passes.mlp_materialization_classifier import MLPMaterializationClassifierPass

        self.passes.append(MLPMaterializationClassifierPass(config))

        if config.compilation_config.pass_config.enable_sp:
            from .passes.sequence_parallelism import SequenceParallelismPass
            from .passes.sequence_parallelism_moe import SequenceParallelismMoePass

            self.passes.append(SequenceParallelismPass(config))
            self.passes.append(SequenceParallelismMoePass(config))
        self._emit_mlp_classifier_event(
            {
                "event": "graph_fusion_configured",
                "manager_instance_id": id(self),
                "passes": [pass_.__class__.__name__ for pass_ in self.passes],
            }
        )
