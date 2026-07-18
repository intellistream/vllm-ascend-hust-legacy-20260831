# KV Cache Block Gather

本文记录 `kv_cache_block_gather` 的一次设备侧搬运流水实验：我们尝试用
`TQueBind` 去掉显式的 UB→UB 拷贝，但实测性能反而下降；随后又对双队列的
`BUFFER_NUM` 做了小范围 sweep。这里把实验、结论和最终选择留在算子旁边，
避免以后仅凭“少一次拷贝一定更快”的直觉重复走一遍。

## 算子的数据路径

算子根据两组 block ID 完成离散 gather：

```text
out[dst_block_ids[i]] = src_pages[src_block_ids[i]]
```

`src_pages` 既可以是设备 GM，也可以是经
`aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED)` 注册后、AI Core 可见的
host 内存。每个 block 会被切成最多 1024 个元素的 tile，由多个 AIV core
按 grid-stride 方式处理。

当前 kernel 使用传统的三段流水：

```text
mapped-host/device GM -> VECIN UB -> VECOUT UB -> device GM
                          CopyIn      local copy      CopyOut
```

对应两个独立队列：

```cpp
TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
```

## 为什么曾经考虑 `TQueBind`

这个 gather 不做 Add、Cast 等数学运算，中间只有：

```cpp
DataCopy(outLocal, inLocal, copyElems);
```

从数据语义看，这次 UB→UB 拷贝确实是恒等操作。因此一个很自然的简化是：

```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
```

让同一个 `LocalTensor` 从 VECIN 直接交给 VECOUT，数据路径缩短为：

```text
mapped-host/device GM -> bound UB -> device GM
```

`TQueBind` 仍然表达 MTE2→MTE3 的事件依赖，但不再分配独立的输入、输出
UB，也没有显式的 local-to-local `DataCopy`。CANN 自带的纯搬运实现中也能
看到这种写法，所以它在语义和 API 使用上都是成立的，不是 correctness
问题。

## 实验方法

使用 [`tools/benchmark_kv_gather_vs_span.py`](../../tools/benchmark_kv_gather_vs_span.py)
测量，主要配置如下：

- Ascend 910B2，CANN 9.0；
- `float16`；
- 512 个 selected blocks，K/V 两个 part；
- block 大小为 4 KiB、16 KiB、64 KiB；
- 每种 block 大小覆盖 8 种 span 碎片程度；
- 每个 case warmup 5 次、正式测量 30 次；
- 每个版本都运行 `span-first` 和 `mapped-first` 两种 backend 顺序；
- host 注册时间排除在 steady-state kernel 时间之外；
- 每个 case 校验完整的非零输出。

下表使用两种 backend 顺序、8 种 mapping 的 **mapped-gather device event
mean** 聚合值。百分比以原双队列 `BUFFER_NUM=1` 为基准；负数表示回退。

## `TQueBind` sweep 结果

| 实现 | `BUFFER_NUM` | 4 KiB event ms | 相对基线 | 16 KiB event ms | 相对基线 | 64 KiB event ms | 相对基线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 双队列 + UB copy | 1 | 0.2670 | 基线 | 0.7998 | 基线 | 2.9343 | 基线 |
| `TQueBind` | 1 | 0.3119 | -14.4% | 0.8332 | -4.0% | 2.9820 | -1.6% |
| `TQueBind` | 2 | 0.2912 | -8.3% | 0.8104 | -1.3% | 2.9566 | -0.8% |
| `TQueBind` | 4 | 0.2870 | -7.0% | 0.8212 | -2.6% | 2.9646 | -1.0% |

结论：

1. `TQueBind<..., 1>` 的确受单 buffer 限制，特别是 4 KiB 小块；
2. `BUFFER_NUM=2` 补回了大部分损失，是最均衡的 `TQueBind` 选择；
3. 增加到 4 没有稳定收益；
4. 即使选择 2，`TQueBind` 仍然没有超过原来的双队列。

wall-clock 聚合值呈现相同方向，因此这个结论不是单纯由 event/wall
计时口径差异造成的。

## 为什么最终不用 `TQueBind`

“少搬一次数据”并不等于“关键路径更短”。`TQue`/`TQueBind` 不只是 UB
内存容器，还描述生产者、消费者之间的流水位置、事件同步和 buffer
生命周期。当前结果说明：

- 独立 VECIN/VECOUT 队列形成的流水，对这个 mapped-host 搬运路径更有利；
- 中间 UB copy 的局部带宽成本很低，可能被其他搬运阶段部分隐藏；
- `TQueBind` 改变了 MTE2→MTE3 的依赖路径，省掉 payload copy 的收益不足以
  抵消新的同步或串行化成本；
- block 越大，mapped-host/GM 外部带宽越占主导，所以两者差距缩小到约 1%。

上面关于流水和事件的解释是由结果推导出的工作假设，还不是 profiler
证明的微架构归因。但“`TQueBind` 在当前目标环境更慢”已经由两种 backend
顺序和完整 correctness 校验重复确认，因此工程决策不需要等待更昂贵的
profiling。

## 双队列 `BUFFER_NUM` sweep

恢复双队列后，又测试了每个队列 1、2、4 个 buffer：

| 双队列 `BUFFER_NUM` | 4 KiB event ms | 相对 1 | 16 KiB event ms | 相对 1 | 64 KiB event ms | 相对 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2670 | 基线 | 0.7998 | 基线 | 2.9343 | 基线 |
| **2** | **0.2653** | **+0.7%** | **0.8007** | **-0.1%** | **2.8886** | **+1.6%** |
| 4 | 0.2914 | -8.4% | 0.8149 | -1.9% | 2.9421 | -0.3% |

`BUFFER_NUM=2` 在 4 KiB 基本持平、16 KiB 属于测量噪声范围，在 64 KiB
则在两种 backend 顺序中都小幅领先。增加到 4 明显没有价值，4 KiB 反而
出现较大回退。因此最终保留：

```cpp
constexpr int32_t BUFFER_NUM = 2;
TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
```

双队列的 payload UB 占用近似为：

```text
2 queues * BUFFER_NUM * tileElems * sizeof(T)
```

`tileElems=1024`、`float16` 时，`BUFFER_NUM=2` 每个 core 使用约 8 KiB
payload UB；相对于本算子的 mapped-host 访问成本，这个空间换取小幅流水收益
是可以接受的。

## 最终决策

- 保留独立 `inputQueue_` 和 `outputQueue_`；
- 保留显式 UB→UB `DataCopy`，不要仅凭代码观感删除；
- 使用 `BUFFER_NUM=2`；
- 不把 `TQueBind` 版本作为性能优化合入；
- 如果未来更换 CANN、SoC 或 tile 策略，应重新测量，而不是把本结论当成所有
  Ascend 平台的永久规律。

本次实验的本地原始 CSV、manifest、日志、源码变体和原实现备份保存在：

```text
branch_development_notes/work/kv-gather-tquebind-20260718-173553/
```

其中 `work/` 是本地实验产物目录，不作为源码发布内容；可复现入口仍是上述
benchmark 脚本。
