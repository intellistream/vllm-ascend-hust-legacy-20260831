# P0 — Latency measurement (B1 vs full-residency baseline)

目标: 给已验证数值正确的 B1 路径补上**性能数据**, 回答"offload 到底慢多少、慢在哪"。
不写新功能、不碰主路径、env-gated、可逆。用数据说话。

## 要测的量
1. 宏观对比 (base 全驻留 vs B1 slots=96):
   - TTFT (time to first token)
   - TPOT (time per output token, decode 稳态)
   - throughput (tokens/s)
2. 微观拆解 (decode 单步):
   - stage(H2D) 时间 / 次 — seam op 内 perf_counter (env-gated)
   - router / mlp 在捕获图内, 留给 msprof / torch_npu profiler (现有 breakdown 工具已覆盖, 不重造)

## 数据源决策
- TTFT/TPOT: 优先用 vLLM offline RequestOutput.metrics (first_token_time/arrival/finished)。
  若该版本未填充 metrics → fallback: SamplingParams 单步 vs N 步差分。
- stage 时间: moe_offload_stage_op.py 内 SEW_SEAM_PROBE 分支加 perf_counter, 打 STAGE_MS 标记。

## 进度
- [x] 核对 RequestOutput.metrics 是否可用 → V1 offline 常不填充, 改用差分法 (T(1)≈TTFT, TPOT=(T(N)-T(1))/(N-1))
- [x] 探针: seam stage 计时 (env-gated SEW_SEAM_PROBE, synchronize 包夹, 打 STAGE_MS)
- [x] 探针: probe harness --latency / --latency-repeats 输出 TTFT/TPOT/DECODE_TPS
- [x] 脚本: run_p0_latency.sh (base + b1_slots96, DEV/MAXTOK/REPS 可配)
- [x] UT 47/47 绿; AST/bash 语法通过
- [x] NPU4 实测 (MAXTOK=32 REPS=5) — 完成
- [x] 拆解结论写回 progress.md (续21) + memory

## 结论 (NPU4, base 全驻留 vs B1 slots=96)
- TTFT 190→319ms (+68%); TPOT 44.5→73.7ms (+66%); decode 22.5→13.6 tok/s (−40%)
- decode staging 双峰: 92% 命中<2ms, 8% miss≥10ms(~90ms/次, sum 6062ms)
- 4 层×mean7.94 = 31.8ms ≈ TPOT delta 29.2ms ⟹ decode 慢几乎全是同步 staging, T_overlap=0
- 本测 ASYNC_LOAD=0 = overlap 可回收的上界基线(worst case)
- 方向: P3 钉死砍 miss 率(抹 6062ms 大头) + P1 overlap(藏 90ms/miss) = 回收 ~25/29ms TPOT 惩罚

## 修正基线 (正确对照): eager 单算子 offload vs B1, CPU offload 大小相同
三跑 HBM 占用: base 全驻留 56.90GB / B1 55.78GB / eager-singleop 55.78GB (B1 与 eager 完全同 footprint)。
slots=96 占用拆解: offload 4 层 slot_bank=3.375GiB + resident 44 层全权重=49.5GiB = 52.875GiB
(全驻留 54GiB, B1 省 1.125GiB = 4层×32逐出×9MiB); host_store CPU=4.5GiB。每专家 9MiB(bf16)。
| 配置 (slots=96,nonres={2,3,4,5}) | TTFT | TPOT | decode tok/s | 图捕获 |
| base 全驻留 (captured)            | 190ms | 44.5ms | 22.5 | 是 |
| **B1 offload (captured)**         | 319ms | **73.7ms** | **13.6** | 是 |
| eager 单算子 offload (enforce_eager) | 233ms | **201.3ms** | **4.97** | 否 |
- **核心结论**: 同 HBM footprint 下, B1 图捕获把 decode 做到比 eager 单算子 offload **快 2.73×**
  (TPOT 201→73.7ms; tok/s 4.97→13.6)。这才是"图兼容 offload"的真实贡献 —— 隔离掉 offload 本身代价。
- eager TPOT 201ms ≫ base 44.5ms: 纯算子逐个 dispatch 的 launch overhead(48 层全 eager), 图捕获塌掉它。
- 三段分解: base 44.5(图捕获 launch 已塌) → +29ms staging(B1) ; eager 201 = 无图捕获 launch 惩罚。
- eager run: 920 条 SEW_PROBE branch=EAGER (n_active=8), 0 CAPTURE_SAFE ⟹ 真 offload 单算子, 非退化。
- TTFT B1(319)>eager(233): B1 prefill 含 seam staging + 首次 captured-graph replay 建立开销; 但 decode 决定吞吐。

## 约束 (用户红线)
- ASCEND_RT_VISIBLE_DEVICES (非 CUDA_)。PY=/root/miniconda3/envs/vllm-hust-dev/bin/python
- 不杀别人的 NPU 进程; 端口 8016 不碰。Model=/data/shared-models/Qwen3-30B-A3B
- 不改 router/top-k/gate/combine; 不动 scheduler/model_runner/token_dispatcher 主路径
