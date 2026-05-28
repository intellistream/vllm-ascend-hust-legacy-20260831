# Research Question Reframing for SEW-Offload

## 1. What We Learned from CCF-A Hardware-Driven Systems Papers

This note summarizes how strong systems and architecture papers frame a research problem when a new hardware capability, device class, or interconnect changes old assumptions. The selected examples come from CCF-A-class venues such as OSDI, SOSP, NSDI, ASPLOS, ISCA, and USENIX ATC.

## 2. Reusable Problem-Definition Pattern

The best hardware-driven systems papers usually do not start with "we support new hardware." They start with a sharper chain:

1. A previously reasonable software abstraction was built around an old hardware assumption.
2. A new device or hardware feature invalidates part of that assumption.
3. The new hardware does not automatically solve the problem; it exposes a new control surface.
4. Existing systems either ignore the control surface or use it through an old abstraction.
5. A new system abstraction is needed to turn that hardware property into performance, isolation, or efficiency.

For SEW-Offload, this pattern is more useful than a generic "MoE offloading on NPU" framing. The paper should argue that GPU-style MoE offloading treats expert residency as a dynamic cache/prefetch problem, while Ascend NPUs add another constraint: execution prefers stable tensor addresses, fixed shapes, graph/static-kernel reuse, and explicitly orchestrated movement. Therefore, MoE offloading on Ascend should be redefined as a static expert-window scheduling problem whose objective is to minimize exposed prefetch stall.

## 3. Representative Paper Patterns

| Paper | Venue | Hardware-driven observation | Problem-definition pattern | Lesson for SEW-Offload |
| --- | --- | --- | --- | --- |
| Dune: Safe User-level Access to Privileged CPU Features | OSDI 2012 | Modern virtualization hardware can safely expose privileged CPU features to user-level software. | Kernel-only access to privileged features was an old boundary; new hardware enables a different OS/application split. | Do not just say "Ascend has graph/static kernels"; define which old software boundary prevents us from using them. |
| Arrakis: The Operating System is the Control Plane | OSDI 2014 | Device virtualization can provide safe direct application access to I/O devices. | Kernel mediation was historically necessary, but new hardware makes the kernel-data-plane assumption too expensive. | Our analogue: GPU-style dynamic expert objects make sense when execution is flexible, but fixed-address NPU execution wants a new data-plane/control-plane split for expert weights. |
| BPFS: A File System for Persistent Memory | SOSP 2009 | Byte-addressable persistent memory changes the block-device assumption. | Existing file systems were optimized for block I/O; persistent memory needs new consistency and update mechanisms. | Ascend offloading should not be phrased as "copy weights faster"; it needs a new persistence/residency abstraction for expert slots. |
| TPP: Transparent Page Placement for CXL-Enabled Tiered Memory | ASPLOS 2023 | CXL enables memory capacity expansion, but introduces heterogeneous latency and bandwidth. | Adding far memory is not enough; page placement becomes the key control problem. | Adding host expert storage is not enough; expert placement and prefetch deadlines become the key control problem. |
| CXL-ANNS | USENIX ATC 2023 | CXL memory can support larger datasets, but far-memory access latency must be co-designed with algorithm behavior. | New capacity devices create performance hazards unless software scheduling matches access locality. | MoE expert offloading needs workload-aware scheduling based on expert locality and token counts, not only LRU hit rate. |
| eRPC | NSDI 2019 | Modern datacenter networks and NICs make software RPC overhead dominate. | Hardware is fast enough that old software layering becomes the bottleneck. | Ascend expert compute can be efficient, so offload orchestration and small phase overhead may become the bottleneck. |
| In-Datacenter Performance Analysis of a TPU | ISCA 2017 | Domain-specific accelerators expose matrix units, memory hierarchy, and deterministic serving behavior unlike CPUs/GPUs. | The right question is not whether the accelerator is faster, but which workloads and bottlenecks match the accelerator design. | We should define Ascend-specific bottlenecks: stable execution entries, graph replay, MTE/movement overlap, and grouped MoE phases. |
| TVM | OSDI 2018 | Deep learning hardware diversity makes hand-written libraries insufficient. | New accelerators require a software stack that exposes scheduling, memory, and tensorization decisions. | SEW-Offload should expose expert residency, prefetch, and phase scheduling as first-class controls, not hide them behind a cache policy. |

## 4. Why the Previous RQ Was Too Weak

The earlier formulation was close to:

> Efficient MoE expert offloading on Ascend NPUs under limited HBM.

This is directionally correct but too broad. It has three reviewer risks:

1. It sounds like a port of GPU MoE offloading to Ascend.
2. It does not clearly distinguish our contribution from existing vLLM Ascend grouped MoE execution.
3. It can be judged as an engineering cache/prefetch policy rather than a systems paper with a new abstraction.

The stronger version should explicitly encode the old assumption, the new hardware control surface, and the performance objective.

It should also be phrased as a problem statement rather than as a solution-seeking "how to" question. In systems papers, the research problem usually reads as a causal claim: existing systems make assumption X, the new hardware invalidates X because of property Y, and the mismatch causes bottleneck Z.

## 5. Proposed Problem Definition

### Main problem statement

> Existing MoE offloading systems treat expert weights as dynamically cached device objects and optimize residency mainly through cache replacement and prefetch prediction. This abstraction is incomplete for Ascend NPUs: once HBM is insufficient to keep all experts resident, dynamic expert loading conflicts with the NPU's preference for stable weight addresses, fixed execution windows, graph/static-kernel reuse, and explicitly scheduled data movement. As a result, offloaded MoE inference either exposes host-to-HBM expert loading on the critical path or gives up the static execution regularity that makes Ascend efficient.

### Short paper version

> GPU-style MoE offloading optimizes which experts are cached, but Ascend MoE offloading also depends on where experts reside, when they are loaded, and whether their loading can be hidden behind static, grouped NPU execution.

### Thesis statement

> The central problem is the mismatch between dynamic expert working sets and static NPU execution windows under limited HBM. SEW-Offload addresses this mismatch by separating logical expert identity from physical HBM slots, scheduling expert prefetch by deadline and token demand, and executing resident experts first so that missing expert loads are overlapped rather than exposed.

## 6. Research Hypothesis

GPU-style MoE offloading treats expert weights as dynamically cached device objects. This abstraction is incomplete for Ascend NPUs because it ignores stable address requirements, graph/static-kernel reuse, and explicit data-movement orchestration. A fixed expert-slot window, combined with deadline-aware prefetching and hit-first phased grouped execution, can reduce exposed offloading stall under HBM constraints without changing router decisions, top-k expert activations, or token-combine semantics.

## 7. Decomposed Problem Claims

### Claim 1: Existing MoE offloading assumes flexible dynamic device residency

Prior systems generally model expert offloading as a cache/prefetch problem: the runtime decides which experts should be resident and which experts should be moved from host memory to device memory. This view is natural on GPU systems where dynamic device objects and stream overlap are the primary control surfaces.

### Claim 2: Ascend exposes a second control surface beyond cache membership

On Ascend NPUs, efficient execution also depends on stable tensor addresses, fixed-shape or low-variant execution windows, graph/static-kernel reuse, and explicitly orchestrated data movement. Therefore, expert offloading cannot be evaluated only by whether an expert is cached; it must also preserve where the expert is placed and when its load completes relative to NPU computation.

### Claim 3: The old abstraction produces exposed prefetch stalls

When expert weights are loaded dynamically without a stable slot/window abstraction, the system must often wait for missing experts before grouped MLP execution can proceed. The visible performance loss is not simply the number of misses, but the portion of host-to-HBM loading time that remains exposed on the inference critical path.

### Claim 4: Correctness requires preserving routing semantics

The system problem is constrained by the requirement that router logits, top-k expert ids, gate weights, token dispatch, and output combine behavior remain unchanged. The paper should not solve the problem by retraining the router, dropping tokens, or changing expert activation.

## 8. Recommended Introduction Logic

The introduction should follow this sequence:

1. MoE models create a memory-capacity problem because total expert weights grow faster than active parameters.
2. GPU offloading handles this with expert caching and prefetching, assuming flexible dynamic weight objects and stream overlap.
3. Ascend NPUs change the problem: efficient execution prefers fixed shapes, stable weight entries, graph/static-kernel reuse, and explicit movement orchestration.
4. vLLM Ascend already solves grouped MoE execution through per-expert token counts; this does not solve offloading because expert weights are still assumed resident.
5. The missing abstraction is an offload-aware static expert window: fixed HBM slots, dynamic expert-to-slot mapping, deadline-aware prefetch, and hit-first phased grouped execution.
6. The optimization target is exposed prefetch stall:

   `T_stall = max(0, T_load_miss - T_overlap)`.

7. SEW-Offload shows how to hide host-to-HBM expert loading behind useful NPU work without modifying the MoE model.

## 9. Reviewer Stress Test

### Likely Reviewer Question 1

"Is this just expert caching with a new name?"

Answer: No. Expert caching chooses what is resident. SEW-Offload additionally fixes where experts can reside, how stable slot addresses are preserved, when prefetch jobs must complete, and how execution is split into grouped phases when misses occur. The metric is exposed stall, not only hit rate.

### Likely Reviewer Question 2

"Does vLLM Ascend already solve this with per-expert counts?"

Answer: No. vLLM Ascend's per-expert count/grouped execution represents token dispatch for resident weights. SEW-Offload starts after that point: it manages HBM-limited expert weight residency, stable slot mapping, prefetch, replacement, and phased execution.

### Likely Reviewer Question 3

"Why is this Ascend-specific?"

Answer: The design is motivated by Ascend-friendly execution properties: stable tensor addresses, graph/static-kernel reuse, fixed window variants, explicit data movement, and avoidance of many small per-expert launches. A GPU implementation may adopt similar ideas, but the paper's control surface and evaluation should be written around Ascend-specific constraints and opportunities.

### Likely Reviewer Question 4

"What is the measurable win?"

Answer: The paper should report HBM savings and latency/throughput, but the core mechanism must be evaluated by hidden vs. exposed host-to-HBM load time, copy/compute overlap, slot churn, and P95/P99 latency under controlled slot budgets.

## 10. Sources Used

- Dune: Safe User-level Access to Privileged CPU Features, OSDI 2012. https://www.usenix.org/conference/osdi12/technical-sessions/presentation/belay
- Arrakis: The Operating System is the Control Plane, OSDI 2014. https://www.usenix.org/conference/osdi14/technical-sessions/presentation/peter
- BPFS: A File System for Persistent Memory, SOSP 2009. https://dl.acm.org/doi/10.1145/1629575.1629589
- TPP: Transparent Page Placement for CXL-Enabled Tiered Memory, ASPLOS 2023. https://dl.acm.org/doi/10.1145/3582016.3582034
- CXL-ANNS: Software-Hardware Collaborative Memory Disaggregation and Computation for Billion-Scale Approximate Nearest Neighbor Search, USENIX ATC 2023. https://www.usenix.org/conference/atc23/presentation/chen-jianjun
- eRPC: Fast RPCs for Datacenter Networks, NSDI 2019. https://www.usenix.org/conference/nsdi19/presentation/kalia
- In-Datacenter Performance Analysis of a Tensor Processing Unit, ISCA 2017. https://dl.acm.org/doi/10.1145/3079856.3080246
- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning, OSDI 2018. https://www.usenix.org/conference/osdi18/presentation/chen
