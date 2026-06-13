# vLLM-Ascend 原生不支持 MoE 专家权重卸载推理

本目录存放一份 Beamer 投影片，用来说明官方/原生 vLLM-Ascend 为什么还不支持 MoE 专家权重卸载推理。

编译命令：

```bash
latexmk -xelatex native_moe_offload_evidence.tex
```

说明：

- “原生”指 `/workspace/reference-repos/vllm-ascend` 的官方参考 `main` 分支，编写时检查点为 `7c4ec8e5`。
- `/workspace/vllm-ascend-hust` 的 `research` 分支包含 MoE 卸载原型；它用于说明缺失能力需要在 MoE 执行边界补齐，不等同于官方原生已经支持。
- 本材料讨论的是 MoE 专家权重卸载，不是 KV 缓存 CPU 卸载，也不是 Ascend 的 L2 缓存权重预取。
