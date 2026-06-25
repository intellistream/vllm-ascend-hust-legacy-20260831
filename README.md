<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<div align="center">

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/vllm-project/vllm-ascend)

</div>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>Documentation</b></a> | <a href="https://slack.vllm.ai"><b>#SIG-Ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>Users Forum</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>Weekly Meeting</b></a> |
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2026/05] We released the new official version [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.18.0/) to start using vLLM Ascend Plugin on Ascend.
- [2026/02] We released the new official version [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/12] We released the new official version [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/) to start using vLLM Ascend Plugin on Ascend.
- [2025/09] We released the new official version [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! Please follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) to start deploying large-scale Expert Parallelism (EP) on Ascend.
- [2025/08] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q) with vLLM and Tencent! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/06] [User stories](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html) page is now live! It kicks off with LLaMA-Factory/verl/TRL/GPUStack to demonstrate how vLLM Ascend assists Ascend users in enhancing their experience across fine-tuning, evaluation, reinforcement learning (RL), and deployment scenarios.
- [2025/06] [Contributors](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html) page is now live! All contributions deserve to be recorded, thanks for all contributors.
- [2025/05] We've released the first official version [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)! We collaborated with the vLLM community to publish a blog post sharing our practice: [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html).
- [2025/03] We hosted the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/VtxO9WXa5fC-mKqlxNUJUQ) with vLLM team! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF).
- [2025/02] vLLM community officially created [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) repo for running vLLM seamlessly on the Ascend NPU.
- [2024/12] We are working with the vLLM community to support [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

---

## Overview

vLLM Ascend (`vllm-ascend`) is a community maintained hardware plugin for running vLLM seamlessly on the Ascend NPU.

It is the recommended approach for supporting the Ascend backend within the vLLM community. It adheres to the principles outlined in the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162), providing a hardware-pluggable interface that decouples the integration of the Ascend NPU with vLLM.

By using vLLM Ascend plugin, popular open-source models, including Transformer-like, Mixture-of-Experts (MoE), Embedding, Multi-modal LLMs can run seamlessly on the Ascend NPU.

## About This Fork

This repository is maintained by [vLLM-HUST](https://github.com/vLLM-HUST), focusing on **operator-level optimization** for the vLLM Ascend backend. Our work includes:

- Custom Ascend operator development and optimization (CANN/TIK)
- Performance tuning for attention, MoE, and other critical kernels
- Deep integration with Huawei Ascend hardware features

## Prerequisites

- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series, Atlas 800I A3 Inference series, Atlas A3 Training series, Atlas 300I Duo (Experimental)
- OS: Linux
- Software:
    - Python >= 3.10, < 3.12
    - CANN == 8.5.1 (Ascend HDK version refers to [here](https://www.hiascend.com/document/detail/zh/canncommercial/83RC2/releasenote/releasenote_0000.html))
    - PyTorch == 2.9.0, torch-npu == 2.9.0
    - vLLM (the same version as vllm-ascend)

## Getting Started

Please use the following recommended versions to get started quickly:

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
| v0.19.1rc1 | Latest release candidate | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) for more details |
| v0.18.0 | Latest stable version | See [QuickStart](https://docs.vllm.ai/projects/ascend/en/v0.18.0/quick_start.html) and [Installation](https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html) for more details |

## Research Branch MoE Offload Service

The `research` branch contains the Ascend MoE expert offload prototype. The
validated single-NPU Qwen3-30B-A3B service command is:

```bash
MODEL_PATH=${MODEL_PATH:-/data/shared-models/Qwen3-30B-A3B}
PORT=${PORT:-8016}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6}

ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-30b-a3b \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 536870912 \
  --enforce-eager \
  --ascend-moe-offload-gb 14
```

MoE offload also runs through the legacy eager path above. The `--enforce-eager`
flag dispatches every operator individually, so the runtime CPU decision path of
the offload scheduler is trivially legal — at the cost of paying full per-operator
kernel-launch overhead on every decode step.

### Latest progress: graph-compatible expert offload (B1)

The newest prototype removes the `--enforce-eager` constraint: MoE expert offload
now runs **inside an ACLGraph-captured decode**, while the model still fits in
less HBM than full residency. We call the current milestone **B1**.

**The problem.** Expert offload is data-dependent: which experts to stage into
HBM is only known after the router runs, and that decision needs a device→host
sync (`torch.unique(topk_ids).cpu()`) plus a conditional host→device copy — both
forbidden on a captured stream. Eager mode sidesteps this but loses graph capture
entirely.

**The idea — decouple the control plane from the data plane at a splitting seam.**
The MoE body is cut into three top-level ops:

```
moe_router_indirect  |  moe_offload_stage  |  moe_mlp
  (captured piece)      (splitting op,        (captured piece)
                         runs EAGER between
                         the two captured pieces)
```

`moe_offload_stage` is registered as a `splitting_op`, so the FX graph splitter
excludes it from capture and it runs **eager between two captured pieces**. It
performs the host decision + synchronous staging of the active experts into a
fixed-address **slot bank**, then writes the logical→physical mapping in place
into a persistent (fixed-address) `log2phy` buffer. The captured `moe_mlp` only
ever reads fixed slot tensors + the fixed buffer, so the graph replays safely
while the *contents* change every step. Expert weights for offloaded layers live
on CPU (host store) and are streamed into the `num_slots` HBM slots on demand.

**Correctness (Qwen3-30B-A3B, single NPU, `num_slots=96 < 128` experts, offload
layers {2,3,4,5}).** Output tokens and the full per-position top-20 logprobs are
**bit-identical to the full-residency baseline to 1e-5** — the offload path
changes only *where expert weights live and how they are accessed*, never the
router / top-k / gate / combine semantics. The slot bank holds 96 experts while
the host store holds all 128, so 32 experts per offloaded layer never reside in
HBM (genuine `num_slots < n`).

**Performance — same offload footprint (55.78 GB HBM), only variable is graph
capture.** Compared against the eager single-op offload baseline at identical
`num_slots=96`, prompt, and concurrency=1:

| Config (slots=96, 32 tokens) | TTFT (prefill) | TPOT (per decode step) | Decode tok/s |
|------------------------------|----------------|------------------------|--------------|
| Eager single-op offload      | 242.8 ms       | 212.3 ms               | 4.71         |
| **B1 graph-captured offload**| 308.0 ms       | **69.1 ms**            | **14.5**     |

Graph capture collapses the 48-layer per-operator launch overhead that dominates
eager decode, making **decode 3.07× faster** and the **end-to-end 32-token run
2.79× faster** (2450 ms vs 6824 ms). TTFT is currently higher because prefill has
variable shape and is *not* captured in either config, so B1's prefill only pays
extra staging without a capture benefit — the next milestone (predictive /
overlapped prefetch) targets exactly this.

> Status: research prototype. All offload features are env-gated and default-off;
> the main vLLM Ascend serving paths are unchanged.

### B2: wave-streamed prefill (real HBM savings at a small slot budget)

B1 keeps `num_slots` close to the expert count, so it saves little HBM (offloading
4 layers at `num_slots=96` only frees ~1.1 GB). **B2** pushes `num_slots` far below
the per-call active-expert union (e.g. 8 vs a ~51-expert prefill union) and is the
milestone that delivers real HBM savings.

**The problem.** A prefill forward touches ~51 distinct experts per offloaded
layer, but only `num_slots=8` fit in HBM at once, so a single grouped matmul can't
run — B1 fail-closes (`active expert working set exceeds num_slots`).

**The idea.** Run the prefill MLP in **capacity-bounded waves**. Partition the
active experts into `ceil(N / num_slots)` waves of ≤ `num_slots` each; per wave,
stage that wave's experts into the slot bank, run dispatch → partial grouped
matmul → combine on just those experts (others masked to zero weight), and
accumulate. Because a token's MoE output is a *sum* over its top-k experts and
addition is associative, summing the per-wave contributions reproduces the full
output. Prefill runs eager (it isn't ACLGraph-captured), so the seam defers to the
wave loop there; decode (active set ≤ `num_slots`) keeps the captured single-wave
B1 path, preserving the 3.07× decode speedup.

**End-to-end through the service command.** Set
`VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1` alongside `--ascend-moe-offload-gb` and
**drop `--enforce-eager`**. Autoconfig then derives the slot budget and offloaded
layers, arms the seam + B2, and skips the (non-capturable) PrefetchOffloader.
Validated on Qwen3-30B-A3B, single NPU, `--ascend-moe-offload-gb 14` (12 offloaded
layers, `num_slots=8`), ACLGraph capture on:

| Config | Model weights on HBM | Decode | Output |
|--------|----------------------|--------|--------|
| Full residency (captured) | 56.90 GB | captured | baseline |
| **SEW data plane (B2 + seam)** | **44.24 GB** | **captured** | tokens == baseline |

**12.66 GB (22%) of HBM freed** while keeping graph-captured decode (so the 3×
decode speedup carries over) and never fail-closing on prefill. Output tokens are
identical to the full-residency baseline; top-1 agrees at every position, with
~0.24 nat logprob drift — the expected, harmless cost of re-associating one grouped
matmul into several bf16-summed waves (the algorithm is exact in fp32 / unit tests;
no token flips). The wave executor exposes a two-phase `issue`/`wait` staging
interface (`prefetch_depth`, multi-buffer) so transfer/compute overlap can be added
without changing the planner.

> Status: research prototype. All offload features are env-gated and default-off;
> the main vLLM Ascend serving paths are unchanged.

## Contributing

See [CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html) for more details, which is a step-by-step guide to help you set up the development environment, build and test.

We welcome and value any contributions and collaborations:

- Please let us know if you encounter a bug by [filing an issue](https://github.com/vllm-project/vllm-ascend/issues)
- Please use [User forum](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support) for usage questions and help.

## Branch

vllm-ascend has a main branch and a dev branch.

- **main**: main branch, corresponds to the vLLM main branch, and is continuously monitored for quality through Ascend CI.
- **releases/vX.Y.Z**: development branch, created alongside new releases of vLLM. For example, `releases/v0.13.0` is the dev branch for vLLM `v0.13.0` version.

Below are the maintained branches:

| Branch           | Status       | Note                                 |
|------------------|--------------|--------------------------------------|
| main             | Maintained   | CI commitment for vLLM main branch and vLLM v0.18.0 tag |
| v0.7.1-dev       | Unmaintained | Outdated, no longer maintained. |
| v0.7.3-dev       | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| v0.9.1-dev       | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| v0.11.0-dev      | Unmaintained | Only bug fixes are allowed, and no new release tags anymore. |
| releases/v0.13.0 | Maintained   | CI commitment for vLLM 0.13.0 version |
| releases/v0.18.0 | Maintained   | CI commitment for vLLM 0.18.0 version |
| rfc/feature-name | Maintained   | [Feature branches](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) for collaboration |
  
Please refer to [Versioning policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) for more details.

## Weekly Meeting

- vLLM Ascend Weekly Meeting: <https://tinyurl.com/vllm-ascend-meeting>
- Wednesday, 15:00 - 16:00 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
