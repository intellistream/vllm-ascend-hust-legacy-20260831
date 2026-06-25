#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Decide whether we should enable CP parallelism.
    "VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL", "0"))),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_gmm_combine_decode` / `dispatch_ffn_combine`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # Whether to enable balance scheduling in the v1 scheduler.
    # Platform validation: only PD-mixed mode (`kv_role='kv_both'` or no kv_transfer_config).
    # Not supported in PD-disaggregated mode (`kv_producer` / `kv_consumer` only).
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # Whether to enable utility-based victim selection in scheduler preemption.
    "VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION", "0"))
    ),
    # Emergency kill switch for utility-based victim selection.
    "VLLM_ASCEND_UTILITY_KILL_SWITCH": lambda: bool(int(os.getenv("VLLM_ASCEND_UTILITY_KILL_SWITCH", "0"))),
    # Completion factor weight in utility delta calculation.
    "VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT": lambda: float(
        os.getenv("VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT", "0.5")
    ),
    # Preemption-count factor weight in utility delta calculation.
    "VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT", "0.3")),
    # Minimum KV utilization ratio required to enable utility ranking.
    "VLLM_ASCEND_UTILITY_KV_GATE": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_KV_GATE", "0.0")),
    # Cooldown window (seconds) between two utility-based victim selections.
    "VLLM_ASCEND_UTILITY_COOLDOWN_S": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_COOLDOWN_S", "0.0")),
    # Minimum running queue size required before enabling utility-based victim selection.
    "VLLM_ASCEND_UTILITY_MIN_RUNNING": lambda: int(os.getenv("VLLM_ASCEND_UTILITY_MIN_RUNNING", "1")),
    # Whether to capture shared-snapshot counterfactual records for utility decisions.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED": lambda: bool(
        int(os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED", "0"))
    ),
    # Number of top-ranked candidates to keep in each utility decision snapshot.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K": lambda: int(os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K", "3")),
    # Number of recent utility decision snapshots retained in memory.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE", "32")
    ),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # High-level Ascend MoE expert offload switch. 0 or unset keeps the normal
    # path. A positive value enables the PrefetchOffloader + fixed-slot MoE
    # defaults without using cpu_offload_gb/UVA.
    "VLLM_ASCEND_MOE_OFFLOAD_GB": lambda: float(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_GB", "0")),
    # Enable the Ascend MoE expert offload runtime. Default is disabled.
    "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": lambda: bool(int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "0"))),
    # Trace routed expert working sets without changing execution.
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": lambda: bool(int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY", "0"))),
    # Number of fixed HBM expert slots for later non-trace offload modes.
    "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": lambda: int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "0")),
    # Expert residency policy name for later slot/prefetch modes.
    "VLLM_ASCEND_MOE_OFFLOAD_POLICY": lambda: os.getenv("VLLM_ASCEND_MOE_OFFLOAD_POLICY", "deadline"),
    # Maximum grouped execution phases used by later overlap modes.
    "VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES": lambda: int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_MAX_PHASES", "2")),
    # Enable async host-to-HBM expert loading in later non-trace modes.
    "VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD": lambda: bool(int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_ASYNC_LOAD", "0"))),
    # Bounded in-memory trace history size for MVP-A trace-only mode.
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS": lambda: int(
        os.getenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_MAX_RECORDS", "4096")
    ),
    # Optional JSONL path for cross-process trace-only artifacts.
    "VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH": lambda: os.getenv("VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH", ""),
    # MVP-D.9: comma-separated MoE layer ids that keep full NPU expert weights (no slot path).
    "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS": lambda: os.getenv(
        "VLLM_ASCEND_MOE_OFFLOAD_RESIDENT_LAYER_IDS", ""
    ),
    # MVP-D.9: after release guard is ready, drop original expert Parameter storage on non-resident layers.
    "VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "0"))
    ),
    # MVP-D.10: opt-in dynamic-count layered runtime path selector.
    "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME", "0"))
    ),
    # MVP-D.10: active expert fan-out above this threshold uses full-weight path.
    "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": lambda: int(
        os.getenv("VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD", "0")
    ),
    # MVP-D.11: opt-in post-dispatch phase split semantic prototype.
    # Splits MoE MLP into hit/miss phases. Default off.
    "VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PHASE_SPLIT", "0"))
    ),
    # Option 2: graph-compatible offload (decision/execution decoupling). Default
    # off => current eager-only offload behavior is unchanged.
    "VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE", "0"))
    ),
    # Regime B path ①: insert the vllm::moe_offload_stage splitting op between the
    # router and the grouped MLP so the data-dependent active-set staging runs
    # eager between two captured pieces (per-step, supports num_slots < n). When
    # on, the load-time full-residency hook is skipped for offloaded layers.
    # Default off => Regime A (load-time full residency) behavior is unchanged.
    "VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_STAGE_SEAM", "0"))
    ),
    # Regime B "B2": wave-streamed prefill. When an offloaded layer's eager-prefill
    # active expert union exceeds num_slots, run the MoE MLP in capacity-bounded
    # waves (each <= num_slots experts: stage -> partial grouped matmul -> combine
    # -> accumulate) instead of failing closed. Prefill-only + eager-only; decode
    # still uses the single-wave B1 path. Default off => B1/fail-closed behavior
    # is unchanged.
    "VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_B2_WAVE_PREFILL", "0"))
    ),
    # Autoconfig data-plane selector for --ascend-moe-offload-gb. Default (0) wires
    # the vLLM PrefetchOffloader (eager-only, the legacy service path). When 1,
    # autoconfig instead arms the SEW graph-compatible data plane: it enables
    # GRAPH_COMPATIBLE + STAGE_SEAM + B2_WAVE_PREFILL and does NOT wire
    # PrefetchOffloader (which cannot be ACLGraph-captured on NPU), so the service
    # command can drop --enforce-eager and get captured-decode speedup with B2
    # wave-streamed prefill at a small slot budget.
    "VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE", "0"))
    ),
    # MVP-D.9 verification: optional JSONL path for cross-process profiling artifacts.
    "VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH": lambda: os.getenv("VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH", ""),
    # Non-offload MoE GroupedMatmul trace path. Records grouped dispatch shapes
    # without requiring VLLM_ASCEND_MOE_OFFLOAD_ENABLED.
    "VLLM_ASCEND_MOE_GMM_TRACE_PATH": lambda: os.getenv("VLLM_ASCEND_MOE_GMM_TRACE_PATH", ""),
    # Non-offload MoE GroupedMatmul profile path for fast-path decisions and
    # shape-plan diagnostics.
    "VLLM_ASCEND_MOE_GMM_PROFILE_PATH": lambda: os.getenv("VLLM_ASCEND_MOE_GMM_PROFILE_PATH", ""),
    # Non-offload MoE GroupedMatmul bucket plan path. This path takes
    # precedence over the older compute bucket plan env when both are set.
    "VLLM_ASCEND_MOE_GMM_BUCKET_PLAN_PATH": lambda: os.getenv("VLLM_ASCEND_MOE_GMM_BUCKET_PLAN_PATH", ""),
    # Pipeline-level profiling: record Stage T/R/C/M npu.Event elapsed times (trace-only, no overlap changes).
    "VLLM_ASCEND_MOE_PIPELINE_PROFILING": lambda: bool(
        int(os.getenv("VLLM_ASCEND_MOE_PIPELINE_PROFILING", "0"))
    ),
    # Optional SEW-MoE P1 plan produced by the profiling suite. When set, runtime can classify
    # grouped dispatch signatures before the existing grouped matmul fallback.
    "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": lambda: os.getenv(
        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH", ""

    # -- Sim-LLM: KV reuse optimization ---------------------------------------
    # Whether to enable Sim-LLM KV reuse optimization. When set to 1, the Sim-LLM
    # patch wraps NPUModelRunner.execute_model() at worker init time.
    # 0: disabled (default), 1: enabled.
    "VLLM_ASCEND_SIMLLM_ENABLED": lambda: bool(int(os.getenv("VLLM_ASCEND_SIMLLM_ENABLED", "0"))),

    # Cosine similarity threshold for KV reuse match. Embeddings with cosine
    # similarity >= this value are considered a match. Paper default 0.8.
    # Valid range: [0.0, 1.0]. Higher values = stricter matching, fewer KV reuses.
    "VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD": lambda: float(
        os.getenv("VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD", "0.8")
    ),

    # Number of bits for SimHash LSH projection. More bits = fewer collisions
    # but larger hash storage. Paper default 64 (fits in a single int64).
    # Valid range: [16, 256]. Recommended: 32, 64, or 128.
    "VLLM_ASCEND_SIMLLM_LSH_NUM_BITS": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_LSH_NUM_BITS", "64")
    ),

    # Batch size threshold for switching from exhaustive cosine to LSH bucket
    # merge strategy. Below this threshold: exact cosine per candidate.
    # At or above: LSH bucket membership with KV merging. Default 32.
    "VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD", "32")
    ),

    # Maximum number of cached tasks in KV_Manager. When exceeded, the
    # least-recently-accessed task is evicted (O(1) via OrderedDict).
    # Default 1024. Increase for higher reuse rates on diverse workloads.
    "VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE", "1024")
    ),

    # Number of bottom (early) transformer layers whose KV is retained in the
    # sandwich config for unmatched tasks. Default 3 (layers 0, 1, 2).
    "VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM", "3")
    ),

    # Number of top (late) transformer layers whose KV is retained in the
    # sandwich config for unmatched tasks. Default 3 (layers L-3 .. L-1).
    # Total KV retention = (bottom + top) / num_layers.
    "VLLM_ASCEND_SIMLLM_SANDWICH_TOP": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_SANDWICH_TOP", "3")
    ),

    # Pooling strategy for task embedding extraction from hidden states.
    # Options: "mean" (mean pooling over sequence), "last" (last token only),
    # "cls" (first token / CLS token). Default "mean".
    "VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING": lambda: os.getenv(
        "VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING", "mean"
    ),

    # Batch match ratio threshold for deferral logic. If the fraction of matched
    # tasks in a batch exceeds this value, unmatched tasks are deferred to the
    # next scheduling cycle. Valid range: [0.0, 1.0]. Default 0.5.
    "VLLM_ASCEND_SIMLLM_DEFERRAL_RATIO": lambda: float(
        os.getenv("VLLM_ASCEND_SIMLLM_DEFERRAL_RATIO", "0.5")
    ),

    # Maximum number of times a task can be deferred before being force-processed
    # regardless of match status. Guards against starvation. Default 3.
    "VLLM_ASCEND_SIMLLM_MAX_DEFERRALS": lambda: int(
        os.getenv("VLLM_ASCEND_SIMLLM_MAX_DEFERRALS", "3")
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
