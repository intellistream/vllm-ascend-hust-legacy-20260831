import torch
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU


class AscendNgramProposerNPU(NgramProposerGPU):
    def __init__(self, vllm_config, device: torch.device, runner):
        self.runner = runner
        super().__init__(vllm_config, device=device)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    def propose(
        self,
        num_speculative_tokens: int,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose draft tokens using the PyTorch ngram kernel.

        Scatters newly sampled tokens into the GPU-side token buffer, then
        delegates the n-gram suffix match to the upstream ``NgramGPUKernel``
        which was initialized during ``__init__``.

        Note: The model runner may bypass this method when using the AscendC
        custom op (``npu_ngram_spec_decode``) for better performance on NPU.
        This implementation provides a working fallback via the upstream
        PyTorch kernel so the interface is complete regardless of which
        code path is active at runtime.

        Returns:
            draft_tokens: [batch_size, k] tensor of proposed draft token IDs.
            num_valid_draft_tokens: [batch_size] tensor of valid draft count.
        """
        assert num_speculative_tokens == self.k

        batch_size = num_tokens_no_spec.shape[0]
        max_seq_len = token_ids_gpu.shape[1]
        max_new_tokens = valid_sampled_token_ids_gpu.shape[1]

        # Scatter newly sampled tokens into token_ids_gpu.
        offsets = torch.arange(max_new_tokens, device=self.device)
        write_positions = num_tokens_no_spec.unsqueeze(1) + offsets.unsqueeze(0)
        valid_write_mask = offsets.unsqueeze(0) < valid_sampled_tokens_count.unsqueeze(1)
        in_bounds = write_positions < max_seq_len
        scatter_mask = (
            valid_write_mask & (valid_sampled_token_ids_gpu != -1) & in_bounds
        )
        write_positions_long = write_positions.clamp(max=max_seq_len - 1).long()
        tokens_cast = valid_sampled_token_ids_gpu.to(token_ids_gpu.dtype)
        token_ids_gpu.scatter_(
            1,
            write_positions_long,
            torch.where(scatter_mask, tokens_cast, token_ids_gpu.gather(1, write_positions_long)),
        )

        # Compute updated lengths after scatter.
        clamped_count = valid_sampled_tokens_count.clamp(max=max_seq_len - num_tokens_no_spec)
        updated_lengths = num_tokens_no_spec + clamped_count

        combined_mask = (valid_sampled_tokens_count > 0) & (updated_lengths >= self.min_n)
        return self.kernel(updated_lengths, token_ids_gpu, combined_mask)
