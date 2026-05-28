# SEW-Offload Paper Drafts

本目录当前先承载 SEW-Offload 的论文设计稿。

## 当前文件

- `sew_offload_design.tex`：英文 LaTeX 设计草稿，聚焦整体 Design。

## 当前设计主线

SEW-Offload 的目标不是单纯提高 expert cache hit rate，而是在 Ascend NPU 上把 MoE expert offloading 重新表述为一个可编排、可重叠、可静态化的执行窗口系统：

1. 固定 expert slots 解决 Ascend 能不能稳定执行。
2. deadline-aware prefetch/orchestration 解决 offloading 慢不慢。
3. hit-first phased execution 解决 miss 已经发生时还能不能把等待藏起来。

## 重要边界

- 不重训练 router。
- 不修改 top-k expert 激活。
- 不 drop token。
- 不把 vLLM Ascend 已有的 per-expert count/grouped execution 作为新贡献。
- SEW-Offload 复用现有 grouped MoE 元数据，新增的是 HBM 受限时的固定 slot 驻留、预取编排和分阶段执行。
