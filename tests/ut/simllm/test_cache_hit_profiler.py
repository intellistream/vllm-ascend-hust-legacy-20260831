#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass

import pytest

from vllm_ascend.simllm.cache_hit_profiler import SimLLMCacheHitProfiler


@dataclass
class _Match:
    matched: bool


def test_profiler_accumulates_match_and_rewrite_rates():
    profiler = SimLLMCacheHitProfiler(interval=3)

    assert (
        profiler.record_batch(
            2,
            {_idx: _Match(_idx == 0) for _idx in range(2)},
            1,
            1024,
            2,
            2,
        )
        is None
    )
    snapshot = profiler.record_batch(
        2,
        {_idx: _Match(True) for _idx in range(2)},
        2,
        2048,
        2,
        4,
    )

    assert snapshot is not None
    assert snapshot["queries"] == 4
    assert snapshot["matches"] == 3
    assert snapshot["rewrites"] == 3
    assert snapshot["avg_covered_tokens_per_rewrite"] == 1024
    assert snapshot["cache_size"] == 4


def test_profiler_ignores_decode_only_batches():
    profiler = SimLLMCacheHitProfiler(interval=1)

    assert profiler.record_batch(0, {}, 0, 0, 0, 9) is None
    assert profiler.snapshot()["queries"] == 0


def test_profiler_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="greater than zero"):
        SimLLMCacheHitProfiler(interval=0)
