<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vllm-ascend" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
vLLM Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>关于昇腾</b></a> | <a href="https://docs.vllm.ai/projects/ascend/en/latest/"><b>官方文档</b></a> | <a href="https://slack.vllm.ai"><b>#sig-ascend</b></a> | <a href="https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support"><b>用户论坛</b></a> | <a href="https://tinyurl.com/vllm-ascend-meeting"><b>社区例会</b></a> |
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>

---
*最新消息* 🔥

- [2026/05] 我们发布了新的正式版本 [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2026/02] 我们发布了新的正式版本 [v0.13.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.13.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.13.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2025/12] 我们发布了新的正式版本 [v0.11.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.11.0)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.11.0/)开始在Ascend上部署vLLM Ascend Plugin。
- [2025/09] 我们发布了新的正式版本 [v0.9.1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.9.1)! 请按照[官方指南](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html)开始在Ascend上部署大型专家并行 (EP)。
- [2025/08] 我们与vLLM和腾讯合作举办了[vLLM北京Meetup](https://mp.weixin.qq.com/s/7n8OYNrCC_I9SJaybHA_-Q)，！请在[这里](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)找到演讲材料。
- [2025/06] [用户案例](https://docs.vllm.ai/projects/ascend/en/latest/community/user_stories/index.html)现已上线！展示了LLaMA-Factory/verl/TRL/GPUStack等用户案例，展示了vLLM Ascend如何帮助昇腾用户在模型微调、评估、强化学习 (RL) 以及部署等场景中提升体验。
- [2025/06] [贡献者](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)页面现已上线！所有的贡献都值得被记录，感谢所有的贡献者。
- [2025/05] 我们发布了首个正式版本 [v0.7.3](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3)！我们与 vLLM 社区合作发布了一篇博客文章，分享了我们的实践：[Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html)。
- [2025/03] 我们和vLLM团队举办了[vLLM Beijing Meetup](https://mp.weixin.qq.com/s/CGDuMoB301Uytnrkc2oyjg)! 你可以在[这里](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF)找到演讲材料.
- [2025/02] vLLM社区正式创建了[vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)仓库，让vLLM可以无缝运行在Ascend NPU。
- [2024/12] 我们正在与 vLLM 社区合作，以支持 [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162).

---

## 总览

vLLM 昇腾插件 (`vllm-ascend`) 是一个由社区维护的让vLLM在Ascend NPU无缝运行的后端插件。

此插件是 vLLM 社区中支持昇腾后端的推荐方式。它遵循[[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162)所述原则：通过解耦的方式提供了vLLM对Ascend NPU的支持。

使用 vLLM 昇腾插件，可以让类Transformer、混合专家(MOE)、嵌入、多模态等流行的大语言模型在 Ascend NPU 上无缝运行。

## 关于本分支

本仓库由 [vLLM-HUST](https://github.com/vLLM-HUST) 维护，专注于 **vLLM Ascend 后端的算子级优化**。工作内容包括：

- 自定义 Ascend 算子（CANN/TIK）的开发和优化
- Attention、MoE 等关键算子的性能调优
- 与华为昇腾硬件特性的深度集成

## 准备

- 硬件：Atlas 800I A2 Inference系列、Atlas A2 Training系列、Atlas 800I A3 Inference系列、Atlas A3 Training系列、Atlas 300I Duo（实验性支持）
- 操作系统：Linux
- 软件：
    - Python >= 3.10, < 3.12
    - CANN == 8.5.1 (Ascend HDK 版本参考[这里](https://www.hiascend.com/document/detail/zh/canncommercial/83RC2/releasenote/releasenote_0000.html))
    - PyTorch == 2.9.0, torch-npu == 2.9.0
    - vLLM (与vllm-ascend版本一致)

## 开始使用

推荐您使用以下版本快速开始使用：

| Version    | Release type | Doc                                  |
|------------|--------------|--------------------------------------|
|v0.19.1rc1| 最新RC版本 |请查看[快速开始](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)和[安装指南](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)了解更多|
|v0.18.0| 最新正式/稳定版本 |[快速开始](https://docs.vllm.ai/projects/ascend/en/v0.18.0/quick_start.html) and [安装指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html)了解更多|

## Research 分支：MoE 专家 Offload 服务

`research` 分支包含昇腾 MoE 专家 offload 原型。已验证的单卡 Qwen3-30B-A3B 服务启动命令：

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

上面这条是早期的 **eager 路径** offload：`--enforce-eager` 让每个算子逐个下发，offload
调度器的"运行时 CPU 决策"因此天然合法——代价是每个 decode 步都要承担算子逐个下发的
kernel launch 开销。

### 最新进展：图兼容专家 Offload（B1）

最新原型去掉了 `--enforce-eager` 约束：MoE 专家 offload 现在可以**在被 ACLGraph 捕获的
decode 内运行**，同时模型占用的 HBM 仍小于全量驻留。当前里程碑称为 **B1**。

**问题所在。** 专家 offload 是数据相关的：要把哪些专家搬进 HBM，只有在 router 算完后才知道，
而这个决策需要一次设备→主机同步（`torch.unique(topk_ids).cpu()`）加一次条件性的主机→设备
搬运——这两件事在被捕获的流上都是被禁止的。eager 模式绕开了它，却也彻底失去了图捕获。

**思路——在一个 splitting seam 处把控制面与数据面解耦。** 把 MoE 主体切成三个顶层算子：

```
moe_router_indirect  |  moe_offload_stage  |  moe_mlp
  (被捕获的 piece)       (splitting 算子,      (被捕获的 piece)
                          在两个被捕获 piece
                          之间 EAGER 执行)
```

`moe_offload_stage` 被注册为 `splitting_op`，FX 切图器因此把它排除在捕获区外、让它在
**两个被捕获 piece 之间 eager 执行**。它完成主机决策 + 把本步 active 专家同步 staging 进
**固定地址的 slot bank**，再把"逻辑→物理"映射就地写入一块持久（地址固定）的 `log2phy`
缓冲区。被捕获的 `moe_mlp` 始终只读固定 slot 张量 + 这块固定缓冲区，所以图能安全 replay，
而其**内容**每步都在变。offload 层的专家权重常驻 CPU（host store），按需流式搬进
`num_slots` 个 HBM 槽位。

**正确性（Qwen3-30B-A3B，单卡，`num_slots=96 < 128` 个专家，offload 层 {2,3,4,5}）。**
输出 token 与每个位置的全部 top-20 logprob **与全量驻留基线逐位相同到 1e-5**——offload 路径
只改变*专家权重住在哪、怎么访问*，绝不改 router / top-k / gate / combine 语义。slot bank
持 96 个专家、host store 持全部 128 个，因此每个 offload 层有 32 个专家从不驻留 HBM
（真正的 `num_slots < n`）。

**性能——offload footprint 完全相同（55.78 GB HBM），唯一变量是图捕获。** 在相同的
`num_slots=96`、相同输入、并发=1 下，对比 eager 单算子 offload 基线：

| 配置（slots=96, 32 token） | TTFT（prefill） | TPOT（单步 decode） | Decode tok/s |
|---------------------------|-----------------|---------------------|--------------|
| eager 单算子 offload       | 242.8 ms        | 212.3 ms            | 4.71         |
| **B1 图捕获 offload**      | 308.0 ms        | **69.1 ms**         | **14.5**     |

图捕获把主导 eager decode 的"48 层算子逐个下发"开销整个塌掉，使 **decode 快 3.07×**、
**32 token 端到端快 2.79×**（2450 ms vs 6824 ms）。当前 TTFT 偏高，是因为 prefill 形状可变、
两种配置下都*未被捕获*，所以 B1 的 prefill 只多付了 staging 开销却拿不到捕获补偿——下一个
里程碑（预测式 / 重叠预取）正是冲着这一点去的。

> 状态：研究原型。所有 offload 特性均由环境变量门控、默认关闭；vLLM Ascend 主服务路径不受影响。

### B2：分波流式 prefill（小槽预算下的真实显存节省）

B1 的 `num_slots` 接近专家数，省的显存有限（4 层 offload、`num_slots=96` 只释放约 1.1 GB）。
**B2** 把 `num_slots` 压到远小于单次 active 专家并集（如 8 vs prefill 的 ~51），是带来真实
显存节省的里程碑。

**问题。** 一次 prefill 前向每个 offload 层用到 ~51 个不同专家，但 HBM 一次只放得下
`num_slots=8` 个，单次 grouped matmul 跑不了——B1 直接 fail-close
（`active expert working set exceeds num_slots`）。

**思路。** 把 prefill 的 MLP 跑成**容量分波**：把 active 专家切成 `ceil(N / num_slots)` 波、
每波 ≤ `num_slots`；每波把本波专家 stage 进 slot bank，只对这些专家跑 dispatch → 局部 grouped
matmul → combine（其余掩码为 0 权重），再累加。因为一个 token 的 MoE 输出是其 top-k 专家的
**求和**、加法可结合，所以把各波贡献相加 == 完整输出。prefill 跑 eager（不被 ACLGraph 捕获），
seam 在此 defer 给波循环；decode（active ≤ `num_slots`）仍走被捕获的单波 B1 路径，保住 3.07×
decode 提速。

**端到端走服务命令。** 在 `--ascend-moe-offload-gb` 旁设 `VLLM_ASCEND_MOE_OFFLOAD_SEW_DATAPLANE=1`
并**去掉 `--enforce-eager`**。autoconfig 随即推导槽预算与 offload 层、arm seam+B2、并跳过
（不可捕获的）PrefetchOffloader。在 Qwen3-30B-A3B、单卡、`--ascend-moe-offload-gb 14`（12 个
offload 层、`num_slots=8`）、开 ACLGraph 捕获下验证：

| 配置 | HBM 上模型权重 | Decode | 输出 |
|------|---------------|--------|------|
| 全量驻留（captured） | 56.90 GB | captured | 基线 |
| **SEW 数据面（B2 + seam）** | **44.24 GB** | **captured** | tokens == 基线 |

**释放 12.66 GB（22%）显存**，同时保住图捕获 decode（3× decode 提速继承）、prefill 永不
fail-close。输出 token 与全量驻留基线逐位一致；每个位置 top-1 一致，logprob 漂移 ~0.24 nat
——这是把单次 grouped matmul 重结合成多波 bf16 求和的**预期且无害**代价（算法在 fp32/单测中
精确，无 token 翻转）。波执行器暴露两段式 `issue`/`wait` staging 接口（`prefetch_depth`、多缓冲），
便于后续加搬运/计算重叠而不改 planner。

> 状态：研究原型。所有 offload 特性均由环境变量门控、默认关闭；vLLM Ascend 主服务路径不受影响。

## 贡献

请参考[CONTRIBUTING](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/contribution/index.html)文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作：

- 请通过[Issue](https://github.com/vllm-project/vllm-ascend/issues)来告知我们您遇到的任何Bug。
- 请通过[用户论坛](https://discuss.vllm.ai/c/hardware-support/vllm-ascend-support)来交流使用问题和寻求帮助。

## 分支策略

vllm-ascend有主干分支和开发分支。

- **main**: 主干分支，与vLLM的主干分支对应，并通过昇腾CI持续进行质量看护。
- **releases/vX.Y.Z**: 开发分支，随vLLM部分新版本发布而创建，比如`releases/v0.13.0`是vllm-ascend针对vLLM `v0.13.0` 版本的开发分支。

下面是维护中的分支：

| 分支              | 状态         | 备注                  |
|------------------|--------------|----------------------|
| main             | Maintained   | 基于vLLM main分支和vLLM最新版本（v0.18.0）CI看护   |
| v0.7.1-dev       | Unmaintained | 不再维护 |
| v0.7.3-dev       | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| v0.9.1-dev       | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| v0.11.0-dev      | Unmaintained | 只允许Bug修复，不会再发布新版本 |
| releases/v0.13.0 | Maintained   | 基于vLLM v0.13.0版本CI看护 |
| releases/v0.18.0 | Maintained   | 基于vLLM v0.18.0版本CI看护 |
| rfc/feature-name | Maintained   | 为协作创建的[特性分支](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#feature-branches) |

请参阅[版本策略](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)了解更多详细信息。

## 社区例会

- vLLM Ascend 每周社区例会: <https://tinyurl.com/vllm-ascend-meeting>
- 每周三下午，15:00 - 16:00 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

## 许可证

Apache 许可证 2.0，如 [LICENSE](./LICENSE) 文件中所示。
