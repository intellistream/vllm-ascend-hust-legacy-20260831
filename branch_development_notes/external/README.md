# External Experiment Tools

This directory is a local-only landing zone for branch-specific experiment
tools. Do not add these tools as git submodules and do not commit their source
trees into this repository.

## vllm-hust-perf-analyzer

Purpose:

- offline TraceLoom analysis for Ascend/CANN `msprof` outputs;
- post-processing support for device KV gather and CPU-offload experiments;
- readable reports over profiler SQLite databases.

Manual checkout:

```bash
cd branch_development_notes/external
git clone https://github.com/vLLM-HUST/vllm-hust-perf-analyzer.git
cd vllm-hust-perf-analyzer
git rev-parse HEAD
```

When an experiment depends on a specific version, record the commit SHA in the
run's `observations.md` or in the relevant work summary under
`branch_development_notes/work`.

Typical local use:

```bash
cd branch_development_notes/external/vllm-hust-perf-analyzer
python3 -m pip install -e .
traceloom analyze /path/to/msprof_output
```

Smoke-tested local use without installation:

```bash
cd branch_development_notes/external/vllm-hust-perf-analyzer
PYTHONPATH=. python3 -m traceloom analysis \
  examples/kickstart_smoke/msprof_raw \
  --out-dir /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/work/traceloom-kickstart-full-20260619 \
  --output-mode bundle \
  --max-main-events-per-device 0
```

The full-analysis smoke test was run at commit
`4f47a3f502916340dd74c40fc94ef1be8a1cf38c`. The generated augmented DB files
are large derived artifacts and are ignored under `branch_development_notes/work`.

Notes:

- The checkout is ignored by `branch_development_notes/external/.gitignore`.
- The source of truth for the tool remains its upstream repository.
- This branch records only the tool URL, purpose, and pinned commit used by a
  specific experiment.
