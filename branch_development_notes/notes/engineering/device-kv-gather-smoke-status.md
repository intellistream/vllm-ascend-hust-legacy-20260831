# Device KV Gather Smoke Status

Generated on 2026-06-18.

## Short Answer

Yes. After extracting the experimental custom op from `tmp/cann-stack` into the
main repository custom-op build tree, the Docker route can now reproduce the
prototype end to end on this machine.

The current reproducible state is:

```text
Docker build: pass
Python op registration: pass
Custom ACLNN opapi packaging: pass
Direct gather execution: pass
```

## Current Machine State

Branch:

```text
experiment/device-kv-gather
```

Hardware:

- `npu-smi info` sees 8 Ascend 910B2 devices.
- Devices 1 and 4 are currently mostly occupied by `VLLMEngineCor` processes.
- Other devices appear available from `npu-smi`.

System/runtime:

- `/usr/bin/python3` exists.
- `torch`, `torch_npu`, and `vllm` are not importable from the current Python.
- `vllm_ascend` imports only as the local source package.
- No local virtualenv/conda Python was found under `/home/jingyuan` in the
  quick search.
- No built `*_C_ascend*.so` was found under the repo, `/home/jingyuan`, or
  `/tmp`.

CANN:

- `/usr/local/Ascend` exists.
- Toolkit version visible under `/usr/local/Ascend/ascend-toolkit/8.2.RC1`.
- Driver/version info reports `25.2.1`.
- Kernel is Linux `5.15.0-25-generic` on `aarch64`.
- `libascendcl.so` exists.
- A quick search did not find system `libopapi.so` or `libcust_opapi.so`.

## Existing Source Support

The branch contains the prototype pieces:

- `csrc/torch_binding.cpp`
  - registers `torch.ops._C_ascend.kv_cache_block_gather`
  - calls `aclrtHostRegister(... ACL_HOST_REGISTER_MAPPED ...)`
  - dynamically resolves:
    - `aclnnKvCacheBlockGatherGetWorkspaceSize`
    - `aclnnKvCacheBlockGather`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
  - optionally uses `kv_cache_block_gather`
  - falls back to tensor copy when unavailable or unsupported
- `vllm_ascend/envs.py`
  - defines:
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER`
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_LIB`
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`

The branch notes already recorded that no focused unit/system test was added
for this op.

## Smoke Coverage Before This Check

No committed smoke test for `kv_cache_block_gather` was found.

The existing branch note explicitly said:

```text
the mapped-host gather path still needs NPU/CANN validation
```

So the answer to "did we ensure smoke succeeds?" is:

```text
No, not yet.
```

## Smoke Harness Added

Added a direct op smoke script:

```text
tools/smoke_device_kv_gather.py
```

It:

- imports `torch` and `torch_npu`
- loads `vllm_ascend.vllm_ascend_C` or an explicit `--op-lib`
- checks that `torch.ops._C_ascend.kv_cache_block_gather` is registered
- creates a CPU source KV tensor
- creates an NPU destination KV tensor
- gathers selected source blocks into selected destination blocks
- synchronizes NPU
- compares output against CPU expected data

The script compiles with `py_compile` in the current environment, but cannot be
executed here until `torch`/`torch_npu` and the built extension are available.

Example command once the environment is ready:

```bash
python3 tools/smoke_device_kv_gather.py --device npu:0 --dtype float16
```

If the extension is not importable as `vllm_ascend.vllm_ascend_C`, pass it
explicitly:

```bash
python3 tools/smoke_device_kv_gather.py \
  --op-lib /path/to/vllm_ascend_C*.so \
  --device npu:0 \
  --dtype float16
```

If a custom opapi library is needed:

```bash
python3 tools/smoke_device_kv_gather.py \
  --op-lib /path/to/vllm_ascend_C*.so \
  --opapi-lib /path/to/libcust_opapi.so \
  --device npu:0
```

## Docker Reproduction Check

Checked on 2026-06-18 with:

```text
quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler
```

Container setup:

- launched a disposable privileged container
- mounted `/dev/davinci0`, `/dev/davinci_manager`, `/dev/devmm_svm`,
  `/dev/hisi_hdc`
- mounted host Ascend driver and `npu-smi`
- mounted this repo read-only, copied it to `/tmp/vllm-ascend-hust` inside the
  container, then built from the copied tree
- sourced `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- set `ASCEND_RT_VISIBLE_DEVICES=0` and `ASCEND_VISIBLE_DEVICES=0`

The container environment can see NPU:

```text
torch_npu import: ok
torch.npu.is_available(): True
npu-smi: visible
```

Important shell note: for the existing long-running workspace containers,
`docker exec ... bash -lc ...` can hang in login-shell initialization. Use
`/bin/sh -c ...` for simple probes.

Editable build result from this branch:

```text
branch: experiment/device-kv-gather
commit: 968cc21f
pip install -e . --no-build-isolation -v: pass
vllm_ascend_C.cpython-311-aarch64-linux-gnu.so: built
torch.ops._C_ascend.kv_cache_block_gather: registered
```

Registration probe:

```python
import torch
import vllm_ascend.vllm_ascend_C
print(hasattr(torch.ops._C_ascend, "kv_cache_block_gather"))
```

Result:

```text
has gather True
```

Direct smoke command:

```bash
python3 tools/smoke_device_kv_gather.py --device npu:0 --dtype float16
```

Result:

```text
RuntimeError: aclnnKvCacheBlockGather or
aclnnKvCacheBlockGatherGetWorkspaceSize not found in op_api libraries
```

The checked image contains CANN 8.5.1 opapi libraries:

```text
/usr/local/Ascend/cann-8.5.1/aarch64-linux/lib64/libopapi.so
/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
```

But:

```bash
strings libopapi.so | grep -E 'aclnnKvCacheBlockGather|KvCacheBlockGather'
```

finds no matching symbol in the checked libraries.

This matches the prototype implementation in `csrc/torch_binding.cpp`: the
Python op is registered by vLLM Ascend, but actual execution dynamically
resolves the ACLNN gather functions at runtime via `GetOpApiFuncAddr` or an
optional `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB` override.

## Custom Op Inventory Check

Initial inventory showed that the gather ACLNN API was not present in the
in-repo custom ACLNN op inventory.

Evidence:

- `branch_development_notes/branch-notes.md` records only the torch binding and
  CPU offload connector changes for `kv_cache_block_gather`; it does not record
  a new `csrc/<op>/op_host` or `csrc/<op>/op_kernel` implementation.
- `git diff main...HEAD --name-status` shows no new CANN op directory for
  gather. The branch-specific source change is `csrc/torch_binding.cpp`.
- `rg 'KvCacheBlockGather|kv_cache_block_gather' csrc` finds references in
  `csrc/torch_binding.cpp`, but no standalone custom op source tree.
- `csrc/build_aclnn.sh` does not include `kv_cache_block_gather` or
  `block_gather` in the 910B/910C `CUSTOM_OPS` lists.
- The checked source tree has only `vllm_ascend/_cann_ops_custom/.gitkeep`; no
  prebuilt custom opapi library is committed.

The prototype initially had this shape:

```text
torch binding/wrapper: present
host memory mapping logic: present
CPU offload connector integration: present
in-repo ACLNN custom op implementation: absent
prebuilt custom opapi library: absent
```

That explained the earlier Docker result: the extension could register
`torch.ops._C_ascend.kv_cache_block_gather`, but the first execution fails when
the wrapper tries to resolve the actual ACLNN entry points.

## Custom Op Integration

The experimental implementation was found under:

```text
tmp/cann-stack/custom-ops/kv_cache_block_gather
```

It was extracted into the main repository under:

```text
csrc/kv_cache_block_gather/
```

Integrated files:

- `csrc/kv_cache_block_gather/op_host/CMakeLists.txt`
- `csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_compat.h`
- `csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_def.cpp`
- `csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_graph_plugin.cpp`
- `csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_infershape.cpp`
- `csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_tiling.cpp`
- `csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather.cpp`
- `csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather.h`
- `csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather_tiling_data.h`
- `csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather_tiling_key.h`

Main build-chain change:

```text
csrc/build_aclnn.sh now includes kv_cache_block_gather in the 910B and 910C
custom-op lists.
```

The extracted source needed light adaptation for this repository's custom-op
build conventions:

- local include paths instead of `examples/custom_ops/...`
- `log/ops_log.h` instead of the standalone-stack `log/log.h`
- a small compatibility header for missing local logging/check macros
- main-repo targets: `op_host_aclnn`, `optiling`, `opsproto`, and
  `opmaster_ct` where applicable

Single-op Docker build command:

```bash
cd /tmp/vllm-ascend-hust/csrc
bash build.sh -n kv_cache_block_gather -c ascend910b
```

Single-op build result:

```text
CANN-custom_ops--linux.aarch64.run: generated
packages/vendors/vllm-ascend/op_api/include/aclnn_kv_cache_block_gather.h: generated
packages/vendors/vllm-ascend/op_api/lib/libcust_opapi.so: generated
```

Symbol check:

```text
0000000000001bd0 T aclnnKvCacheBlockGather
00000000000016e0 T aclnnKvCacheBlockGatherGetWorkspaceSize
```

## Latest Docker Reproduction Result

Checked after integration on 2026-06-18 with:

```text
quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler
```

Full editable build command shape:

```bash
python3 -m pip install -e . --no-build-isolation -v
```

Result:

```text
pip install -e . --no-build-isolation -v: pass
vllm_ascend/_cann_ops_custom/vendors/vllm-ascend: installed
vllm_ascend_C.cpython-311-aarch64-linux-gnu.so: built
```

Custom op files in the installed package include:

```text
op_api/include/aclnn_kv_cache_block_gather.h
op_api/lib/libcust_opapi.so
op_impl/ai_core/tbe/kernel/ascend910b/kv_cache_block_gather/*.o
op_impl/ai_core/tbe/kernel/ascend910b/kv_cache_block_gather/*.json
op_impl/ai_core/tbe/vllm-ascend_impl/dynamic/kv_cache_block_gather.py
```

Runtime environment used for the smoke:

```bash
CUSTOM_OPP=/tmp/vllm-ascend-hust/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
export ASCEND_CUSTOM_OPP_PATH=$CUSTOM_OPP
export LD_LIBRARY_PATH=$CUSTOM_OPP/op_api/lib:${LD_LIBRARY_PATH:-}
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB=$CUSTOM_OPP/op_api/lib/libcust_opapi.so
```

Installed opapi symbol check:

```text
0000000000047e70 T aclnnKvCacheBlockGather
0000000000047980 T aclnnKvCacheBlockGatherGetWorkspaceSize
```

Python registration check:

```text
has gather True
```

Direct smoke command:

```bash
python3 tools/smoke_device_kv_gather.py --device npu:0 --dtype float16
```

Result:

```text
PASS: kv_cache_block_gather smoke succeeded
device=npu:0 dtype=float16 src_shape=(8, 16, 2, 32)
```

Remaining caveat:

```text
The smoke validates the direct prototype op path in Docker on one NPU. It does
not yet validate long-running CPU offload integration, scheduler interaction,
request cancellation, or host-memory lifecycle safety.
```

To make the smoke green, we need one of:

1. add the missing custom ACLNN op implementation under `csrc/`, add it to
   `csrc/build_aclnn.sh`, rebuild, and ensure the generated custom opapi library
   exports `aclnnKvCacheBlockGather*`;
2. provide an external custom opapi library and point
   `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB` at it;
3. retarget the wrapper to an existing available ACLNN/custom backend.

## Minimum Repro Checklist

Before claiming stable reproduction on this machine:

1. Activate or create a Python environment with:
   - `torch`
   - `torch_npu`
   - `vllm`
   - editable or installed `vllm_ascend`
2. Source CANN environment if needed:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. Build vllm-ascend extension from this branch.
4. Confirm:

```python
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C
assert hasattr(torch.ops._C_ascend, "kv_cache_block_gather")
```

5. Run direct op smoke:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python3 tools/smoke_device_kv_gather.py --device npu:0 --dtype float16
```

6. Repeat for:
   - `float16`
   - `bfloat16`
   - `float32`
   - several block orders
   - at least one long repeated loop

7. Only after direct op smoke succeeds, run an end-to-end CPU offload case with:

```bash
VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1
```

## Current Risk To Reproducibility

- Current host Python environment is missing required packages.
- No built extension is present in the host source tree.
- Docker can build the extension, but the checked CANN 8.5.1 opapi libraries do
  not expose `aclnnKvCacheBlockGather`.
- The prototype dynamically resolves gather ACLNN symbols at runtime; source
  compilation alone does not prove the platform provides those symbols.
- The prototype registers host memory but has no unregister path.
- The old CPU offload connector path is deprecated and not the intended
  production integration point.

## Verdict

Docker reproduction is viable for build and registration validation, and should
be the default way to reproduce this prototype on this machine.

The direct gather smoke is not green yet. The blocker is not the local build
chain; it is the absence of the required ACLNN runtime symbols in the checked
CANN/opapi libraries and in the current in-repo custom op inventory. To make
the smoke pass, one of these must be true:

- add or locate a compatible custom opapi implementation that provides
  `aclnnKvCacheBlockGather` and
  `aclnnKvCacheBlockGatherGetWorkspaceSize`
- provide that compatible custom opapi library through
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`
- change the prototype to call a different available backend
