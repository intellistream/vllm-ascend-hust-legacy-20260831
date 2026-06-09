import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))



class TestNPUModelRunnerDPSyncBuffers(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.dp_size = 2
        runner.dp_rank = 1
        runner.vllm_config = MagicMock()
        return runner

    @patch(
        "vllm_ascend.worker.model_runner_v1."
        "should_skip_allreduce_across_dp_group"
    )
    def test_sync_metadata_reuses_padding_buffer_when_allreduce_skipped(
        self, mock_should_skip_allreduce
    ):
        mock_should_skip_allreduce.return_value = True
        runner = self._build_runner()

        _, first_tokens_after_padding, _ = runner._sync_metadata_across_dp(7)
        _, second_tokens_after_padding, _ = runner._sync_metadata_across_dp(9)

        self.assertIsNotNone(first_tokens_after_padding)
        self.assertIsNotNone(second_tokens_after_padding)
        self.assertEqual(
            first_tokens_after_padding.data_ptr(),
            second_tokens_after_padding.data_ptr(),
        )
        self.assertEqual(second_tokens_after_padding.tolist(), [9, 9])

    @patch("vllm_ascend.worker.model_runner_v1.get_dp_group")
    @patch("vllm_ascend.worker.model_runner_v1.dist.all_reduce")
    @patch(
        "vllm_ascend.worker.model_runner_v1."
        "should_skip_allreduce_across_dp_group"
    )
    def test_sync_metadata_reuses_buffers_for_allreduce_path(
        self,
        mock_should_skip_allreduce,
        mock_all_reduce,
        mock_get_dp_group,
    ):
        mock_should_skip_allreduce.return_value = False
        mock_get_dp_group.return_value = SimpleNamespace(cpu_group=object())
        all_reduce_buffer_ptrs = []

        def fake_all_reduce(packed_tensor, group):
            del group
            all_reduce_buffer_ptrs.append(packed_tensor.data_ptr())
            if len(all_reduce_buffer_ptrs) == 1:
                packed_tensor[0].copy_(torch.tensor([5, 11], dtype=torch.int32))
                packed_tensor[1].fill_(CUDAGraphMode.FULL.value)
            else:
                packed_tensor[0].copy_(torch.tensor([13, 3], dtype=torch.int32))
                packed_tensor[1].fill_(CUDAGraphMode.PIECEWISE.value)

        mock_all_reduce.side_effect = fake_all_reduce
        runner = self._build_runner()

        first_max_tokens, first_tokens_after_padding, first_mode = (
            runner._sync_metadata_across_dp(7)
        )
        second_max_tokens, second_tokens_after_padding, second_mode = (
            runner._sync_metadata_across_dp(2)
        )

        self.assertEqual(first_max_tokens, 11)
        self.assertEqual(second_max_tokens, 13)
        self.assertEqual(first_mode, CUDAGraphMode.FULL)
        self.assertEqual(second_mode, CUDAGraphMode.PIECEWISE)
        self.assertEqual(all_reduce_buffer_ptrs[0], all_reduce_buffer_ptrs[1])
        self.assertIsNotNone(first_tokens_after_padding)
        self.assertIsNotNone(second_tokens_after_padding)
        self.assertEqual(
            first_tokens_after_padding.data_ptr(),
            second_tokens_after_padding.data_ptr(),
        )
        self.assertEqual(second_tokens_after_padding.tolist(), [13, 3])


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])


if __name__ == "__main__":
    unittest.main()
