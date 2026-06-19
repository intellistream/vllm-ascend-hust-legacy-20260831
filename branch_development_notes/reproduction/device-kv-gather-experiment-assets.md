# Device KV Gather Experiment Assets

Generated: 2026-06-19 Asia/Shanghai

This note records the repository-local assets for reproducing the experimental
device KV gather work. Experimental matrices, runners, and result files are
kept under `branch_development_notes` so the main source tree only carries the
minimal source-level changes needed for the prototype.

## Asset Map

| Asset | Purpose |
| --- | --- |
| `branch_development_notes/benchmarks/device_kv_gather/matrix.json` | Phase-1 raw microbenchmark matrix. |
| `branch_development_notes/tools/bench_device_kv_gather_matrix.py` | Runs the compiled C++ raw benchmark across the matrix and writes manifest, JSONL, CSV, and Markdown output. |
| `branch_development_notes/tools/bench_cpu_npu_offload_transfer.py` | Runs synthetic worker-local H2D, D2H, and bidirectional copy baselines through `CpuNpuOffloadingHandler`. |
| `branch_development_notes/work/raw-matrix-quick-20260619-073434` | First completed quick raw matrix run. |
| `branch_development_notes/work/worker-local-transfer-smoke-cmake-20260619-084437` | First completed worker-local H2D/D2H/bidirectional copy smoke. |
| `branch_development_notes/work/worker-local-transfer-quick-20260619-084951` | First completed worker-local copy quick baseline matrix. |
| `branch_development_notes/work/experiment-matrix-gap-report.md` | Capability gap and progress report for the experiment matrix. |
| `branch_development_notes/notes/reproduction.md` | Base custom-op build and smoke reproduction guide. |

## Raw Matrix Reproduction

Use the same Docker shape as `branch_development_notes/notes/reproduction.md`.
The benchmark binary must be compiled inside an environment where the custom
op package has been built.

High-level flow:

```bash
REPO=/home/jingyuan/workspace/vllm-ascend-hust
IMAGE=quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler

docker run --rm --privileged --network=host --ipc=host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v "$REPO":/workspace/vllm-ascend-hust:rw \
  -w /tmp \
  "$IMAGE" \
  /bin/bash -lc '
set -euxo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ]; then
  source /usr/local/Ascend/cann-8.5.1/set_env.sh
fi

export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0

rm -rf /tmp/vllm-ascend-hust
cp -a /workspace/vllm-ascend-hust /tmp/vllm-ascend-hust
cd /tmp/vllm-ascend-hust
git config --global --add safe.directory /tmp/vllm-ascend-hust

cd csrc
bash build.sh -n kv_cache_block_gather -c ascend910b
CUSTOM_OPP=$(find /tmp/vllm-ascend-hust/csrc/build/_CPack_Packages -path "*/packages/vendors/vllm-ascend" -type d | head -1)
export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_OPP
export LD_LIBRARY_PATH=$CUSTOM_OPP/op_api/lib:${LD_LIBRARY_PATH:-}
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB=$CUSTOM_OPP/op_api/lib/libcust_opapi.so

g++ -std=c++17 \
  /tmp/vllm-ascend-hust/csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp \
  -I/usr/local/Ascend/ascend-toolkit/latest/include \
  -I$CUSTOM_OPP/op_api/include \
  -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
  -L$CUSTOM_OPP/op_api/lib \
  -lascendcl -lcust_opapi \
  -Wl,-rpath,$CUSTOM_OPP/op_api/lib \
  -o /tmp/kv_cache_block_gather_benchmark

python3 /tmp/vllm-ascend-hust/branch_development_notes/tools/bench_device_kv_gather_matrix.py \
  --binary /tmp/kv_cache_block_gather_benchmark \
  --matrix /tmp/vllm-ascend-hust/branch_development_notes/benchmarks/device_kv_gather/matrix.json \
  --warmup 1 \
  --iters 3 \
  --output-dir /workspace/vllm-ascend-hust/branch_development_notes/work/raw-matrix-manual
'
```

For threshold-quality data, rerun with `--iters 10` or `--iters 30` and a new
output directory under `branch_development_notes/work`.

## Worker-Local Copy Baseline Reproduction

The worker-local transfer runner uses
`torch.ops._C_ascend.swap_blocks_batch`. The base Docker image may not have this
branch's extension registered, so build the current checkout first:

```bash
python3 -m pip install -e . --no-build-isolation -v
```

As of 2026-06-19, `tmp/cann-stack` has been removed from the repository index
and must not be used as a reproduction resource. For transfer-path validation,
the full ACLNN custom-op set and bundled AscendC kernel target are not required.
Use the narrow 910B build path below to build only the
`kv_cache_block_gather` ACLNN op plus the `vllm_ascend_C` torch binding:

```bash
cd /tmp/vllm-ascend-hust

VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather \
VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0 \
SOC_VERSION=ascend910b1 \
python3 -m pip install -e . --no-build-isolation -v

PYTHONPATH=/tmp/vllm-ascend-hust python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C
print("has swap_blocks_batch", hasattr(torch.ops._C_ascend, "swap_blocks_batch"))
print("has kv_cache_block_gather", hasattr(torch.ops._C_ascend, "kv_cache_block_gather"))
PY
```

If editable install is not desirable, the equivalent manual CMake path can build
only `vllm_ascend_C`:

```bash
cd /tmp/vllm-ascend-hust

PYBIND11_CMAKE=$(python3 -m pybind11 --cmakedir)
TORCH_CMAKE=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
TORCH_NPU_PATH=$(python3 -m pip show torch-npu | sed -n 's/^Location: //p')/torch_npu

cmake -S . -B /tmp/vllm-ascend-hust-cmake-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest} \
  -DPYTHON_EXECUTABLE=$(command -v python3) \
  -DPYTHON_INCLUDE_PATH=$PY_INC \
  -DCMAKE_INSTALL_PREFIX=/tmp/vllm-ascend-hust/vllm_ascend \
  -DCMAKE_PREFIX_PATH=$PYBIND11_CMAKE\;$TORCH_CMAKE \
  -DSOC_VERSION=ascend910b1 \
  -DTORCH_NPU_PATH=$TORCH_NPU_PATH \
  -DVLLM_ASCEND_BUILD_ASCENDC_KERNELS=OFF

cmake --build /tmp/vllm-ascend-hust-cmake-build -j 8 --target vllm_ascend_C
cmake --install /tmp/vllm-ascend-hust-cmake-build

PYTHONPATH=/tmp/vllm-ascend-hust python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C
print("has swap_blocks_batch", hasattr(torch.ops._C_ascend, "swap_blocks_batch"))
print("has kv_cache_block_gather", hasattr(torch.ops._C_ascend, "kv_cache_block_gather"))
PY
```

Then run:

```bash
PYTHONPATH=/tmp/vllm-ascend-hust \
python3 branch_development_notes/tools/bench_cpu_npu_offload_transfer.py \
  --device-id 0 \
  --block-bytes 4096 16384 65536 \
  --selected-blocks 8 32 128 512 \
  --directions h2d d2h bidirectional \
  --patterns random \
  --warmup 3 \
  --iters 10 \
  --output-dir branch_development_notes/work/worker-local-transfer-baseline
```

This establishes the copy baseline before adding mapped-host gather backend
selection to `CpuNpuOffloadingHandler`.

## Worker-Local Mapped H2D Reproduction Status

The worker-local runner now has a `--h2d-backend mapped` mode, which sets
`VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1` before constructing
`CpuNpuOffloadingHandler`.

As of 2026-06-19, the mapped H2D smoke has not yet produced a valid transfer
row. The implementation call shape has been corrected to match the registered
op signature:

```text
torch.ops._C_ascend.kv_cache_block_gather(
  src_block_ids,
  src_pages,
  dst_block_ids,
  out,
)
```

The remaining validation item is to rerun mapped H2D using the narrow 910B build
path above:

- single-op `kv_cache_block_gather` builds and exports the ACLNN symbols;
- `VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather` keeps the ACLNN build focused on
  the required op;
- `VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0` skips unrelated bundled AscendC kernel
  targets while preserving `vllm_ascend_C` registration for both transfer ops.

The latest failed mapped smoke record is:

```text
branch_development_notes/work/worker-local-mapped-h2d-smoke-20260619-095730
```

Use a 910B build that registers both `torch.ops._C_ascend.swap_blocks_batch`
and `torch.ops._C_ascend.kv_cache_block_gather` before rerunning:

```bash
PYTHONPATH=/tmp/vllm-ascend-hust \
python3 branch_development_notes/tools/bench_cpu_npu_offload_transfer.py \
  --device-id 0 \
  --h2d-backend mapped \
  --block-bytes 4096 \
  --selected-blocks 8 \
  --directions h2d \
  --patterns random \
  --warmup 1 \
  --iters 3 \
  --limit 1 \
  --output-dir branch_development_notes/work/worker-local-mapped-h2d-smoke
```

## Current Known Validation State

- Raw custom-op build: verified with the single-op build flow.
- Raw C++ benchmark compile: verified in Docker.
- Raw matrix quick run: verified, 52 cases / 208 rows, all pass.
- Worker-local transfer runner syntax and dry-run: verified.
- Worker-local transfer real smoke: verified through manual CMake build of
  `vllm_ascend_C`; 4KB x 8 blocks H2D/D2H/bidirectional all pass.
- Worker-local transfer quick baseline: verified through manual CMake build of
  `vllm_ascend_C`; 18 cases / 30 rows, all pass.
- Worker-local mapped H2D selector: implemented experimentally and call-order
  fixed; real smoke should be rerun with the narrow 910B transfer build above.
- `tmp/cann-stack` has been removed from the repository index and is no longer
  part of the reproduction path.

## Optional Trace Analysis Tool

For profiler post-processing, use `vllm-hust-perf-analyzer` as a manual,
local-only checkout:

```bash
cd branch_development_notes/external
git clone https://github.com/vLLM-HUST/vllm-hust-perf-analyzer.git
cd vllm-hust-perf-analyzer
git rev-parse HEAD
```

Do not add it as a git submodule. The checkout is ignored locally by
`branch_development_notes/external/.gitignore`. Record the commit SHA in the
specific experiment run that uses TraceLoom output.

The local kickstart trial was run at commit
`4f47a3f502916340dd74c40fc94ef1be8a1cf38c` and recorded in:

```text
branch_development_notes/work/traceloom-kickstart-full-20260619/observations.md
```

For comparable full-analysis records, run TraceLoom with:

```bash
PYTHONPATH=. python3 -m traceloom analysis /path/to/msprof_raw \
  --out-dir /workspace/vllm-ascend-hust/branch_development_notes/work/traceloom-<run-id> \
  --output-mode bundle \
  --max-main-events-per-device 0
```

Use TraceLoom output to compare matched copy-backend and mapped-gather `msprof`
runs at the execution-structure level: loop shape, copy/gather placement,
communication time, idle attribution, and cross-rank imbalance.
