# Native MoE Offload Evidence Slides

This folder contains a Beamer slide deck explaining why upstream/native
vLLM-Ascend does not support MoE expert offload inference.

Build:

```bash
latexmk -xelatex native_moe_offload_evidence.tex
```

Scope:

- "Native" refers to the official/reference vLLM-Ascend main branch checked at
  `/workspace/reference-repos/vllm-ascend` (`7c4ec8e5` during authoring).
- The local research branch at `/workspace/vllm-ascend-hust` already contains a
  prototype MoE offload service. The slides use it only as evidence that the
  missing native capability has to be added around the MoE execution boundary.
- The deck distinguishes MoE expert weight offload from KV cache CPU offload
  and Ascend cache/L2 weight prefetch.
