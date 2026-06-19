# TraceLoom Analysis Bundle

- msprof_input: `/home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/external/vllm-hust-perf-analyzer/examples/kickstart_smoke/msprof_raw`
- output_mode: `bundle`

## Primary Outputs

- `db01.traceloom_augmented.db`: copied msprof SQLite DB with TraceLoom `traceloom_*` tables and views.
- `db02.traceloom_augmented.db`: copied msprof SQLite DB with TraceLoom `traceloom_*` tables and views.
- `summary.md`: run-level device and loop summary.
- `tree-map.md`: readable node-cost map; copy a `node` id into SQL drill-down queries.
- `meta.json`: analyzer parameters and generated file paths.
- `queries/*.sql`: starter SQL reports for the augmented DBs.

## Common Commands

```bash
python3 -m pip install -e .
traceloom analyze /path/to/msprof_output
traceloom report /path/to/msprof_output/traceloom/db01.traceloom_augmented.db \
  --sql /path/to/msprof_output/traceloom/queries/repeat-overview.sql \
  --format md
```

Run `traceloom analyze <msprof_dir> --output-mode full` to also export the legacy CSV/JSON debug tables.

## Query Scripts

- `queries/anchor-aux.sql`
- `queries/node-cost-breakdown.sql`
- `queries/node-events.sql`
- `queries/node-occurrences.sql`
- `queries/repeat-children.sql`
- `queries/repeat-overview.sql`
- `queries/tree-map.sql`
