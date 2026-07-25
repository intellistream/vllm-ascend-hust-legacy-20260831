import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 257, 257), torch.bfloat16),
        ((7, 13, 17), torch.float16),
        ((129, 257, 65), torch.bfloat16),
        ((257, 129, 257), torch.float32),
        ((513, 1025, 129), torch.float16),
        ((1025, 257, 257), torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_linear_persistent_overwrites_full_output(shape, dtype):
    """Cover decode, prefill, tail blocks, and every supported dtype."""
    init_device_properties_triton()
    m, n, k = shape
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtype, device="npu")
    weight = torch.randn((n, k), dtype=dtype, device="npu")
    expected = F.linear(x, weight)

    outputs = [linear_persistent(x, weight) for _ in range(3)]
    rtol, atol = (5e-3, 5e-3) if dtype == torch.float32 else (2e-2, 2e-2)

    for output in outputs:
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])
