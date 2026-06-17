import json

import pytest
import torch

from vllm_ascend.ops.sparse_linear import (
    _custom_op_enabled,
    _CUSTOM_OP_MARKED,
    _record_custom_op_invocation,
    activation_sparse_linear,
    activation_sparse_linear_direct,
    activation_sparse_linear_direct_t,
    activation_sparse_linear_ref,
    activation_sparse_linear_packed_t_ref,
    activation_sparse_linear_packed_ref,
    activation_sparse_pack_ref,
)


@pytest.mark.parametrize("inclusive", [False, True])
def test_activation_sparse_linear_cpu_ref_scalar_threshold(inclusive):
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    threshold = torch.tensor(0.4, dtype=torch.float32)

    out = activation_sparse_linear_ref(
        x,
        weight,
        threshold,
        inclusive=inclusive,
    )
    compare = torch.ge if inclusive else torch.gt
    sparse_x = torch.where(compare(x.abs(), threshold), x, torch.zeros_like(x))

    assert torch.allclose(out, sparse_x @ weight.t())


def test_activation_sparse_linear_cpu_ref_row_threshold():
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    threshold = torch.tensor([0.2, 1.0], dtype=torch.float32)

    out = activation_sparse_linear_ref(x, weight, threshold)
    sparse_x = torch.where(
        x.abs() > threshold.reshape(2, 1),
        x,
        torch.zeros_like(x),
    )

    assert torch.allclose(out, sparse_x @ weight.t())


def test_activation_sparse_pack_ref_contract():
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    threshold = torch.tensor([0.2, 1.0], dtype=torch.float32)

    values, indices, counts = activation_sparse_pack_ref(x, threshold)

    assert counts.tolist() == [2, 2]
    assert torch.allclose(values[0, :2], torch.tensor([-0.7, 0.4]))
    assert indices[0, :2].tolist() == [1, 2]
    assert torch.allclose(values[1, :2], torch.tensor([1.2, -1.5]))
    assert indices[1, :2].tolist() == [0, 2]


def test_activation_sparse_linear_packed_ref_matches_masked_linear():
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    threshold = torch.tensor([0.2, 1.0], dtype=torch.float32)

    values, indices, counts = activation_sparse_pack_ref(x, threshold)
    packed = activation_sparse_linear_packed_ref(values, indices, counts, weight)
    masked = activation_sparse_linear_ref(x, weight, threshold)

    assert torch.allclose(packed, masked)


def test_activation_sparse_linear_packed_t_ref_matches_masked_linear():
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    threshold = torch.tensor([0.2, 1.0], dtype=torch.float32)

    values, indices, counts = activation_sparse_pack_ref(x, threshold)
    packed_t = activation_sparse_linear_packed_t_ref(
        values,
        indices,
        counts,
        weight.t().contiguous(),
    )
    masked = activation_sparse_linear_ref(x, weight, threshold)

    assert torch.allclose(packed_t, masked)


def test_activation_sparse_linear_cpu_wrapper_matches_masked_linear():
    x = torch.tensor(
        [[0.1, -0.7, 0.4], [1.2, -0.2, -1.5]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    threshold = torch.tensor([0.2, 1.0], dtype=torch.float32)

    packed_wrapper = activation_sparse_linear(x, weight, threshold)
    packed_t_wrapper = activation_sparse_linear(
        x,
        weight,
        threshold,
        weight_t=weight.t().contiguous(),
    )
    direct_wrapper = activation_sparse_linear_direct(x, weight, threshold)
    direct_t_wrapper = activation_sparse_linear_direct_t(
        x,
        weight.t().contiguous(),
        threshold,
    )
    masked = activation_sparse_linear_ref(x, weight, threshold)

    assert torch.allclose(packed_wrapper, masked)
    assert torch.allclose(packed_t_wrapper, masked)
    assert torch.allclose(direct_wrapper, masked)
    assert torch.allclose(direct_t_wrapper, masked)


def test_activation_sparse_custom_op_marker_writes_once(tmp_path, monkeypatch):
    marker_path = tmp_path / "custom_op_marker.jsonl"
    monkeypatch.setenv("VLLM_ASCEND_SPARSE_LINEAR_MARKER_PATH", str(marker_path))
    _CUSTOM_OP_MARKED.clear()

    _record_custom_op_invocation(
        "activation_sparse_linear_packed_t",
        {
            "x_shape": [2, 3],
            "threshold_numel": 2,
            "inclusive": True,
            "weight_t_provided": True,
        },
    )
    _record_custom_op_invocation(
        "activation_sparse_linear_packed_t",
        {"x_shape": [4, 5]},
    )
    _record_custom_op_invocation(
        "activation_sparse_linear",
        {
            "x_shape": [1, 3],
            "threshold_numel": 1,
            "inclusive": False,
        },
    )
    _record_custom_op_invocation(
        "activation_sparse_linear_direct_t",
        {
            "x_shape": [1, 3],
            "threshold_numel": 1,
            "inclusive": False,
        },
    )

    records = [
        json.loads(line)
        for line in marker_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert records[0]["op"] == "activation_sparse_linear_packed_t"
    assert records[0]["x_shape"] == [2, 3]
    assert records[0]["threshold_numel"] == 2
    assert records[0]["inclusive"] is True
    assert records[0]["weight_t_provided"] is True
    assert records[1]["op"] == "activation_sparse_linear"
    assert records[1]["x_shape"] == [1, 3]
    assert records[1]["threshold_numel"] == 1
    assert records[1]["inclusive"] is False
    assert records[2]["op"] == "activation_sparse_linear_direct_t"
    assert records[2]["x_shape"] == [1, 3]
    assert records[2]["threshold_numel"] == 1
    assert records[2]["inclusive"] is False


def _requires_npu_custom_op():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for the AscendC sparse kernel")
    if not _custom_op_enabled():
        pytest.skip("Ascend custom ops must be enabled")


def _npu_sparse_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-1, 5e-2
    return 5e-2, 5e-2


@pytest.mark.parametrize("dtype", [torch.float16])
def test_activation_sparse_linear_npu_matches_ref(dtype):
    _requires_npu_custom_op()
    torch.manual_seed(0)
    x = torch.randn(3, 32, device="npu", dtype=dtype)
    weight = torch.randn(17, 32, device="npu", dtype=dtype)
    threshold = torch.tensor([0.25, 0.5, 0.75], device="npu")

    actual = activation_sparse_linear(x, weight, threshold)
    direct = activation_sparse_linear_direct(x, weight, threshold)
    direct_t = activation_sparse_linear_direct_t(
        x,
        weight.t().contiguous(),
        threshold,
    )
    expected = activation_sparse_linear_ref(x, weight, threshold)
    atol, rtol = _npu_sparse_tolerances(dtype)

    assert torch.allclose(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(direct.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(direct_t.cpu(), expected.cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("inclusive", [False, True])
def test_activation_sparse_linear_npu_packed_ops_edge_cases(dtype, inclusive):
    _requires_npu_custom_op()
    x = torch.tensor(
        [
            [0.1, -0.2, 0.3, -0.4, 0.5],
            [0.25, -0.75, 1.0, -1.25, 1.5],
            [2.0, -2.5, 3.0, -3.5, 4.0],
        ],
        device="npu",
        dtype=dtype,
    )
    # The first row has nnz=0. The second row exercises equality with
    # inclusive=True. Output dim 17 exercises the packed_t tail tile.
    threshold = torch.tensor([10.0, 1.0, 2.25], device="npu", dtype=torch.float32)
    weight = torch.randn(17, 5, device="npu", dtype=dtype)
    weight_t = weight.t().contiguous()

    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold.contiguous(),
        inclusive,
    )
    packed = torch.ops._C_ascend.activation_sparse_linear_packed(
        values,
        indices,
        counts,
        weight.contiguous(),
    )
    packed_t = torch.ops._C_ascend.activation_sparse_linear_packed_t(
        values,
        indices,
        counts,
        weight_t,
    )
    direct_t = torch.ops._C_ascend.activation_sparse_linear_direct_t(
        x.contiguous(),
        weight_t,
        threshold.contiguous(),
        inclusive,
    )
    wrapper = activation_sparse_linear(
        x,
        weight,
        threshold,
        inclusive=inclusive,
        weight_t=weight_t,
    )
    expected = activation_sparse_linear_ref(
        x,
        weight,
        threshold,
        inclusive=inclusive,
    )

    expected_counts = [0, 3 if inclusive else 2, 4]
    atol, rtol = _npu_sparse_tolerances(dtype)
    assert counts.cpu().tolist() == expected_counts
    assert torch.allclose(packed.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(packed_t.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(direct_t.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(wrapper.cpu(), expected.cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_activation_sparse_linear_npu_batched_row_topk_matches_ref(dtype):
    _requires_npu_custom_op()
    torch.manual_seed(1)
    batch_size = 8
    input_dim = 128
    output_dim = 1153
    keep = input_dim // 2
    x = torch.randn(batch_size, input_dim, device="npu", dtype=dtype)
    weight = torch.randn(output_dim, input_dim, device="npu", dtype=dtype)
    weight_t = weight.t().contiguous()
    topk_values, _ = torch.topk(x.abs().to(dtype=torch.float32), keep, dim=-1)
    threshold = topk_values[..., -1].contiguous()

    values, indices, counts = torch.ops._C_ascend.activation_sparse_pack(
        x.contiguous(),
        threshold,
        True,
    )
    packed_t = torch.ops._C_ascend.activation_sparse_linear_packed_t(
        values,
        indices,
        counts,
        weight_t,
    )
    direct_t = torch.ops._C_ascend.activation_sparse_linear_direct_t(
        x.contiguous(),
        weight_t,
        threshold,
        True,
    )
    wrapper = activation_sparse_linear(
        x,
        weight,
        threshold,
        inclusive=True,
        weight_t=weight_t,
    )
    expected = activation_sparse_linear_ref(
        x,
        weight,
        threshold,
        inclusive=True,
    )

    assert counts.min().item() >= keep
    assert counts.max().item() <= input_dim
    atol, rtol = _npu_sparse_tolerances(dtype)
    assert torch.allclose(packed_t.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(direct_t.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(wrapper.cpu(), expected.cpu(), atol=atol, rtol=rtol)


def test_activation_sparse_linear_npu_bf16_wrapper_falls_back_to_ref():
    _requires_npu_custom_op()
    torch.manual_seed(2)
    x = torch.randn(3, 32, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(17, 32, device="npu", dtype=torch.bfloat16)
    threshold = torch.tensor([0.25, 0.5, 0.75], device="npu")

    actual = activation_sparse_linear(x, weight, threshold)
    direct = activation_sparse_linear_direct(x, weight, threshold)
    expected = activation_sparse_linear_ref(x, weight, threshold)

    atol, rtol = _npu_sparse_tolerances(torch.bfloat16)
    assert torch.allclose(actual.cpu(), expected.cpu(), atol=atol, rtol=rtol)
    assert torch.allclose(direct.cpu(), expected.cpu(), atol=atol, rtol=rtol)
