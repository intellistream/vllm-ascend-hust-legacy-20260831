# TraceLoom Kickstart Trial

Date: 2026-06-19 Asia/Shanghai

Tool:

- repository: `https://github.com/vLLM-HUST/vllm-hust-perf-analyzer.git`
- local checkout: `branch_development_notes/external/vllm-hust-perf-analyzer`
- commit: `4f47a3f502916340dd74c40fc94ef1be8a1cf38c`

## Command

Run from `branch_development_notes/external/vllm-hust-perf-analyzer`:

```bash
PYTHONPATH=. python3 -m traceloom analysis \
  examples/kickstart_smoke/msprof_raw \
  --out-dir /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/work/traceloom-kickstart-full-20260619 \
  --output-mode bundle \
  --max-main-events-per-device 0
```

`--max-main-events-per-device 0` is important for full analysis. The default
quick path truncates main events at 5000 per device, which is useful for smoke
inspection but changes loop repeat counts and should not be mixed with full
experiment comparisons.

## Result

TraceLoom successfully analyzed the checked-in two-device Ascend `msprof`
sample and produced:

- `summary.md`
- `tree-map.md`
- `meta.json`
- `queries/*.sql`
- `db01.traceloom_augmented.db`
- `db02.traceloom_augmented.db`

The augmented DB files are generated artifacts and are ignored by
`branch_development_notes/work/.gitignore`; rerun the command above to recreate
them when needed.

Key full-analysis summary:

- device 0: 11008 anchors, `used_total_us=2656130.728`
- device 1: 10510 anchors, `used_total_us=1674496.76`
- dominant recovered loop on device 0: `N014 Repeat x36`
- dominant recovered loop on device 1: `N010 Repeat x36`
- inner layer-like repeated block: `Repeat x24` under the outer loop on both
  devices

Representative device 0 high-cost nodes from SQL drill-down:

| node | label | occurrences | total_us |
| --- | --- | ---: | ---: |
| `N020` | `aclnnMm_MatMulCommon_MatMulV2` | 864 | 9993815.485 |
| `N015` | `hcom_allReduce__#_#_#` | 36 | 2359537.309 |
| `N021` | `hcom_allReduce__#_#_#` | 864 | 2015425.221 |
| `N028` | `PpMatmulAccumAtomicKernel` | 1 | 581388.116 |
| `N019` | `AtbRopeKernel` | 864 | 301345.037 |

## Why This Matters For This Branch

This tool is useful because it turns noisy `msprof` SQLite timelines into a
small, reviewable structure:

- it recovers repeated prefill/decode-like loops instead of only reporting flat
  kernel totals;
- it places communication, compute kernels, and idle gaps in the same structural
  tree;
- it preserves SQL drill-down back to concrete profiler events;
- it supports `report --attach`, so later we can compare multiple rank/device
  augmented DBs with custom SQL.

For the mapped-host KV gather experiment, this gives us a bridge from raw
microbenchmark wins to end-to-end runtime evidence. The next useful integration
is to capture `msprof` for matched copy-backend and mapped-gather runs, then
compare:

- whether the decode loop structure stays aligned;
- whether new gather/copy-related events appear inside the critical repeated
  region;
- whether comm/idle proportions shift;
- whether rank-to-rank imbalance increases;
- whether first-token and steady-decode phases show different attribution.

## Notes For Future Use

- Keep the analyzer as a manual local checkout under
  `branch_development_notes/external`; do not add it as a git submodule.
- Record the analyzer commit in every run that depends on it.
- Use full mode for comparable experiment records:
  `--max-main-events-per-device 0`.
- Use quick/default mode only for exploratory smoke checks.
- Avoid naive cross-rank joins on `label` alone because repeated same-label
  nodes can match the wrong structural position. Prefer joins that also include
  depth, repeat count, occurrence count, anchor count, and parent/edge context.
