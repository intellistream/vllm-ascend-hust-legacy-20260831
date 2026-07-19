# Dense BF16 SwiGLU Stage 0 prototype

This directory is an independent, correctness-first prototype for
`down(silu(gate(x)) * up(x))`. It does not call the Triton implementation.

Stage 0 evidence is deliberately limited to a clean host C++ build/test and a
static AscendC ABI/contract audit. The AscendC source has not been compiled with
CANN, launched on an NPU, profiled, or shown to be performant. The finite
schedule space in `contract.json` is frozen by evidence-redesign protocol v2.
That revision and the executable bundle still require three-reviewer exact-SHA
signoff before any reservation request can be generated. After a bound grant,
the exact source must pass CANN custom-op compile and Torch-schema validation
before any NPU kernel, model, or service launch.
