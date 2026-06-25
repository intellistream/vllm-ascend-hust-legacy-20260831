# Benchmark Suite — 4 configs × {conc=1, conc=10}, 严格隔离

目标: 同一 ShareGPT(100条)benchmark 下, 对比 4 配置的 TTFT/TPOT/吞吐/显存, 写入 README。
**严格隔离**: 每配置独立 server 进程, 跑完按 PID 杀 + 确认 NPU 释放, 再起下一个。不同时跑。

## 4 配置 (仅 offload/capture 变量不同; 其余同参考命令)
1. 全驻留 + ACLGraph        : MOE_OFFLOAD_ENABLED=0, 无 --enforce-eager (capture on)
2. 全驻留 + 单算子           : MOE_OFFLOAD_ENABLED=0, --enforce-eager
3. GB=14 单算子 slot=8       : --ascend-moe-offload-gb 14, --enforce-eager (默认 dataplane=PrefetchOffloader; 参考命令原样)
4. GB=14 B2+seam 图捕获 slot=8: --ascend-moe-offload-gb 14, SEW_DATAPLANE=1, 无 --enforce-eager

## 共享 server 参数 (参考命令)
--max-model-len 512 --max-num-seqs 1 --max-num-batched-tokens 512
--kv-cache-memory-bytes 536870912 --dtype bfloat16 -tp 1, port=8020 (绝不碰 8016)

## 客户端
vllm bench serve, ShareGPT 100 prompts, --max-concurrency 1 / 10, --request-rate inf
数据: benchmarks/results/moe_offload_real_sharegpt_qwen3_30b_a3b/ShareGPT_prompt_le256_for_mlen512.json (1000条取100)

## 采集
TTFT(mean/median), TPOT(mean/median), 吞吐(out tok/s + req/s) ← bench JSON
HBM: server 日志 "Loading model weights took X GB" + 服务期 npu-smi 峰值

## 隔离纪律
- 起 server 记 PID; 等 /health; 跑 conc1→conc10; 按 PID kill (绝不 pgrep python3, 防误杀8016);
  等待 + npu-smi 确认 "No process in device" 才进下一配置。
- 挑空闲卡 (避开 8016 卡 + 别人进程卡)。PY=/root/miniconda3/envs/vllm-hust-dev/bin/python

## 进度
- [ ] 验证 vllm bench serve 数据集参数 + 挑卡
- [ ] 写 run_bench_suite.sh (PID 隔离 + HBM 采集)
- [ ] 跑配置1/2/3/4 (各 conc1+conc10), 8 份 JSON
- [ ] 汇总 → README 英/中表格 (TTFT/TPOT/吞吐/显存)
