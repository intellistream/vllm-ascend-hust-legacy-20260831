# ASC-CORE Batch 1 同步处理日志

**分支**: qingwanruojun/sync-asc-core-b1
**日期**: 2026-06-25
**操作人**: qingwanruojun

## 统计

- ✅ 接受: 27/27
- ❌ 跳过: 0
- ⚠️ 有冲突: 12（全部采用上游版本）
- 📄 主要改动文件: `model_runner_v1.py`, `platform.py`, `ascend_config.py`, `mooncake_connector.py`, `config_data.py`

---

## 1. PR #9049 — [BugFix] Use non-device-specific triton pow function

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9049
**合并日期**: 2026-05-16

**改了什么**: 将专属硬件调用改为通用调用

**文件**: `vllm_ascend/worker/v2/sample/penalties.py` 第 25 行

```diff
- pow = triton.language.extra.ascend.libdevice.pow
+ pow = triton.language.extra.libdevice.pow
```

**为什么**: 新版 triton 已内置支持，不再需要 `.ascend` 前缀。不改的话新版 triton 可能找不到这个函数。

**冲突**: 有 — fork 版本仍是旧的 `.ascend` 写法
**处理**: ✅ 接受上游版本

---

## 2. PR #9312 — [Bugfix] ascend_config: use truthy check for enforce_eager

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9312
**合并日期**: 2026-05-19

**改了什么**: 将 `is True` 严格比较改为隐式 truthy 比较

**文件**: `vllm_ascend/ascend_config.py` 第 357 行

```diff
- if vllm_config.model_config.enforce_eager is True:
+ if vllm_config.model_config.enforce_eager:
```

**为什么**: `enforce_eager` 可能是 `True`、`False` 或其他 truthy 值。用 `is True` 只匹配严格布尔值 `True`，可能导致漏判。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 3. PR #9240 — [BugFix] Update cudagraph mode handling for encoder-decoder models

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9240
**合并日期**: 2026-05-19

**改了什么**: 重构 encoder-decoder 模型的 cudagraph 模式回退逻辑

**文件**: `vllm_ascend/platform.py` 第 383-400 行

```diff
-        if model_config and model_config.is_encoder_decoder is True:
-            if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
-                logger.warning(...)
-            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
+        if (
+            model_config
+            and model_config.is_encoder_decoder
+            and compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
+        ):
+            cudagraph_mode = (
+                CUDAGraphMode.PIECEWISE
+                if compilation_config.mode == CompilationMode.VLLM_COMPILE
+                else CUDAGraphMode.NONE
+            )
+            logger.info_once(...)
+            compilation_config.cudagraph_mode = cudagraph_mode
```

**为什么**: 原来只回退到 PIECEWISE，现在根据编译模式分别处理：VLLM_COMPILE → PIECEWISE，其他 → NONE。同时用 `info_once` 代替 `warning` 减少日志刷屏。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 4. PR #9193 — [BugFix] Fix UB overflow after NPUIR upgrade

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9193
**合并日期**: 2026-05-21

**改了什么**: 给 `tl.maximum` 加 `propagate_nan` 参数

**文件**: `vllm_ascend/worker/v2/sample/logprob.py` 第 46 行

```diff
- max_val = tl.max(tl.maximum(logits, max_val))
+ max_val = tl.max(tl.maximum(logits, max_val, propagate_nan=tl.PropagateNan.ALL))
```

**为什么**: NPUIR 升级后 `tl.maximum` 的默认 NaN 传播行为变了，不加参数会导致 UB（未定义行为）溢出。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 5. PR #9456 — [BugFix] Fix Deepseek-V4 async scheduling with MTP

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9456
**合并日期**: 2026-05-22

**改了什么**: 新增 DSA 位置缓冲区 + stream 同步优化 + compress 分支逻辑

**文件**: `vllm_ascend/worker/model_runner_v1.py`

改动 1 — 新增 DSA 缓冲区（第 386-396 行附近）:
```diff
+        # For deepseek-v4 use only
+        self._dsa_positions_cpu_buf = torch.zeros(
+            max_buffer_num_tokens, dtype=torch.int64,
+            pin_memory=self.pin_memory,
+        )
+        self._dsa_positions_np_buf = self._dsa_positions_cpu_buf.numpy()
```

改动 2 — stream 同步优化:
```diff
+        default_stream = torch.npu.current_stream()
         with torch.npu.stream(self.valid_sampled_token_count_copy_stream):  
-            self.valid_sampled_token_count_copy_stream.wait_stream(torch.npu.current_stream())  
+            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  
```

改动 3 — compress 分支新增 DSA 位置计算逻辑
改动 4 — `_build_attention_metadata` 传入 `positions_cpu` 参数
改动 5 — dummy build 和 force_attention 场景重置 `_dsa_positions_cpu_buf`

**为什么**: Deepseek-V4 使用 compress + MTP 时，需要独立的位置缓冲区，异步调度才不会乱。

**冲突**: 3处 — 均为 fork 缺少上游新增的代码段
**处理**: ✅ 接受上游版本

---

## 6. PR #9500 — [BugFix] Fix Deepseek-V4 P/D disaggregation kv_cache_tensor

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9500
**合并日期**: 2026-05-25

**改了什么**: 加空值保护

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`

```diff
             for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
+                if not kv_cache_tensor.shared_by:
+                    continue
                 share_tensor_addr = []
                 share_tensor_stride = []
```

**为什么**: Deepseek-V4 分离式推理时 `kv_cache_tensor.shared_by` 可能为空，不跳过直接访问会崩溃。

**冲突**: 1处 — 此文件在 fork 中不存在（上游新增文件）
**处理**: ✅ 接受上游版本

---

## 7. PR #9510 — [BugFix] Use Compress ratio to avoid obtaining illegal attributes

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9510
**合并日期**: 2026-05-25

**改了什么**: 加 `compress_ratio > 0` 检查

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`

```diff
- if cur_tensor_group_idx:
+ if cur_tensor_group_idx and compress_ratio > 0:
```

**为什么**: 访问属性前先检查压缩比例合法（>0），避免读取非法属性导致崩溃。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 8. PR #8908 — [BugFix] Fix Mooncake Connector MTP accuracy bug

**链接**: https://github.com/vllm-project/vllm-ascend/pull/8908
**合并日期**: 2026-05-28

**改了什么**: 1 行修复

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

```diff
-        原逻辑（略）
+        修复后逻辑
```

**为什么**: 修复 Mooncake 连接器中 MTP 精度问题。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 9. PR #9519 — [BugFix] Fix an error raised by another FIA params check

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9519
**合并日期**: 2026-05-28

**改了什么**: FIA 参数检查修复

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+5/-3)

**为什么**: 另一处 FIA 参数检查的边界条件问题，修复报错。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 10. PR #9656 — [Performance][BugFix] Move copy after draft_forward

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9656
**合并日期**: 2026-05-29

**改了什么**: 调整 `_copy_valid_sampled_token_count` 调用位置

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+2/-1)

```diff
- _copy_valid_sampled_token_count(...)
- draft_forward(...)
+ draft_forward(...)
+ _copy_valid_sampled_token_count(...)
```

**为什么**: 把 copy 操作移到 draft_forward 之后执行，修复 MTP 中的时序问题。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 11. PR #9745 — [BugFix][KV pool] DSv4 fix lookup/load mismatch

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9745
**合并日期**: 2026-05-30

**改了什么**: 重构 DSv4 KV 池的 key 查找/加载逻辑

**文件**:
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

关键改动 — config_data.py: 新增 `cache_family_ratio` 计算
```diff
+        cache_family_ratio = max(infer_cache_family_ratio(cache_family), 1)
+        group_block_size *= cache_family_ratio
```

关键改动 — 返回值按 ratio 对齐:
```diff
-                yield start_idx, end_idx, self._make_key_by_hash(hash_val)
+                start_idx //= cache_family_ratio
+                end_idx //= cache_family_ratio
+                if end_idx <= start_idx:
+                    continue
+                yield (
+                    start_idx, end_idx,
+                    self._make_key_by_hash(hash_val, kv_cache_group_id=..., cache_role=..., cache_family=...),
+                )
```

关键改动 — pool_worker.py: 跳过 null block
```diff
+            request.skip_null_blocks_by_group = self.group_uses_align_state
```

**为什么**: DSv4 KV 池中 cache_family 不同时 block 大小不同，需要按 ratio 对齐才能正确查找/加载。

**冲突**: 2处
**处理**: ✅ 接受上游版本

---

## 12. PR #9771 — [BugFix] Lazy initialize KV store on put

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9771
**合并日期**: 2026-05-30

**改了什么**: KV 存储 get 操作前检查初始化状态

**文件**:
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

```diff
     def get(self, keys, addrs, sizes):
+        if self._lazy_init and not self._store_initialized:
+            logger.error("MooncakeBackend.get called before store initialization")
+            return
+        assert self.store is not None
```

**为什么**: put 已经有懒初始化检查，但 get 没有。在 store 未初始化时调用 get 会直接崩溃。

**冲突**: 1处 — fork 的 mooncake_backend.py 中 get 方法缺少此检查
**处理**: ✅ 接受上游版本

---

## 13. PR #9808 — [BugFix][P/D] Add compress ratio and block_ids cutting

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9808
**合并日期**: 2026-06-01

**改了什么**: Mooncake hybrid connector 加压缩比例 + block_ids 切割

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` (+27/-2)

**为什么**: 与 #9500/#9510 同系列修复，进一步加固 mooncake connector。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 14. PR #9818 — [BugFix] Fix DSA compressed idle dummy graph OOB

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9818
**合并日期**: 2026-06-02

**改了什么**: DSA 越界修复

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+8行)

**为什么**: DSA compressed idle 模式下 dummy graph 访问越界（OOB），加边界检查。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 15. PR #9782 — [BugFix] support HMA in AscendMultiConnector

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9782
**合并日期**: 2026-06-03

**改了什么**: AscendMultiConnector 支持 HMA（异构内存访问）

**文件**: `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py` (+67/-22)

**为什么**: 新增 HMA 支持，提升多连接器场景下的内存访问效率。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 16. PR #10003 — [BugFix] fix dsv4 piecewise scenario

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10003
**合并日期**: 2026-06-04

**改了什么**: 更新 `platform.py` 中 DSv4 piecewise 场景的配置

**文件**: `vllm_ascend/platform.py`

```diff
-        """Apply Ascend-specific defaults. Set sp_min_token_num=1 when enable_sp and not set."""
+        """Apply Ascend-specific defaults."""
+        # Set sp_min_token_num=1 when enable_sp and not set.
```

**为什么**: DSv4 在 piecewise 模式下配置未正确应用。

**冲突**: 1处 — 注释格式不同
**处理**: ✅ 接受上游版本

---

## 17. PR #10008 — [BugFix] cant find num_kv_head from some model config

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10008
**合并日期**: 2026-06-04

**改了什么**: 兼容缺少 `num_kv_head` 的模型配置

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` (+46/-27)

**为什么**: 某些模型（如量化版）配置中不包含 `num_kv_head`，直接读取会报错。

**冲突**: 1处 — fork 和上游的 mooncake_connector.py 差异较大
**处理**: ✅ 接受上游版本

---

## 18. PR #9863 — [BugFix] Fix bug from cudagraph config mode FULL corner case

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9863
**合并日期**: 2026-06-08

**改了什么**: cudagraph FULL 模式边界修复

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+7/-1)

**为什么**: cudagraph FULL 配置模式下某些边界条件会触发错误的行为。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 19. PR #9843 — [BugFix] update discard_request_mask fix stuck chunked PP

**链接**: https://github.com/vllm-project/vllm-ascend/pull/9843
**合并日期**: 2026-06-08

**改了什么**: 更新 discard_request_mask

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+3行)

**为什么**: 修复 chunked 流水线并行中请求丢弃时卡住的问题。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 20. PR #10188 — [BugFix] Fix extra parameter in mamba copy_bufs

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10188
**合并日期**: 2026-06-08

**改了什么**: 修复 mamba 模型 `_get_mamba_copy_bufs()` 传参错误

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+223/-53)

**为什么**: mamba 模型中调用 `_get_mamba_copy_bufs()` 时参数不匹配，导致运行时错误。

**冲突**: 1处
**处理**: ✅ 接受上游版本

---

## 21. PR #10046 — [BugFix] Add env var to control DP metadata all_reduce

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10046
**合并日期**: 2026-06-08

**改了什么**: 新增两个配置项

**文件**:
- `vllm_ascend/ascend_config.py`
- `vllm_ascend/worker/model_runner_v1.py`

```diff
+        # Whether to use NPU device group for DP metadata all_reduce.
+        self.dp_allreduce_on_npu = additional_config.get("dp_allreduce_on_npu", False)
+
+        # Enable optimized reduce sampling scheme
+        self.enable_reduce_sample = additional_config.get("enable_reduce_sample", False)
```

**为什么**: 用户现可通过环境变量控制 DP 数据的 all_reduce 通信方式（NPU 组 vs CPU 组）。

**冲突**: 1处 — fork 的 ascend_config.py 缺少这两个配置项
**处理**: ✅ 接受上游版本

---

## 22. PR #10102 — [BugFix][Refactor] Reduce Mooncake KV cache register regions

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10102
**合并日期**: 2026-06-08

**改了什么**: 减少 Mooncake KV cache 注册区域

**文件**: 4 个文件 (+289/-38)

**为什么**: 适配 sparse C8 场景，简化 KV cache 注册，减少不必要的内存注册。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 23. PR #10019 — [BugFix] Fix PCP handshake port collision in Mooncake

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10019
**合并日期**: 2026-06-08

**改了什么**: 端口冲突修复

**文件**: `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` (+3/-1)

**为什么**: Mooncake 分层 KV 传输中 PCP 握手端口可能冲突，加随机偏移避免碰撞。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 24. PR #10217 — [BugFix] Align AscendStore grouped hash lookup encoding

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10217
**合并日期**: 2026-06-10

**改了什么**: 对齐 AscendStore hash 查找编码

**文件**: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py` (+107/-43)

**为什么**: DSv4 KV 池中 grouped hash 查找编码不一致，导致某些情况下找不到对应的 KV 块。

**冲突**: 1处
**处理**: ✅ 接受上游版本

---

## 25. PR #10205 — [BugFix][Performance] Fix MTP copy sync

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10205
**合并日期**: 2026-06-10

**改了什么**: MTP copy_valid_sampled_token_count 同步修复

**文件**: `vllm_ascend/worker/model_runner_v1.py` (+16/-8)

**为什么**: MTP 场景中 copy_valid_sampled_token_count 的同步逻辑有问题，可能导致数据不一致。

**冲突**: 无
**处理**: ✅ 自动合并成功

---

## 26. PR #10177 — [BugFix] Hybrid Mamba Attn in kv pooling and kv p2p

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10177
**合并日期**: 2026-06-16

**改了什么**: Hybrid Mamba Attention 在 KV 池化和 P2P 传输中的修复

**文件**:
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`
- `vllm_ascend/patch/platform/patch_mamba_config.py`

**为什么**: Mamba + Attention 混合模型在 KV 池化和 P2P 传输中有兼容性问题。

**冲突**: 1处 — patch_mamba_config.py 差异
**处理**: ✅ 接受上游版本

---

## 27. PR #10567 — [BugFix] Reduce the number of capture sizes on 950

**链接**: https://github.com/vllm-project/vllm-ascend/pull/10567
**合并日期**: 2026-06-17

**改了什么**: 减少 950 上的 graph capture size

**文件**: `vllm_ascend/platform.py` (+142/-39)

**为什么**: Ascend 950 上 capture size 过多导致显存占用高，减少不必要的 capture 优化性能。

**冲突**: 1处
**处理**: ✅ 接受上游版本
