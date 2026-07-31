from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
META_BINDING = REPO_ROOT / "csrc" / "torch_binding_meta.cpp"


def _function_body(source: str, function_name: str) -> str:
    start = source.index(function_name)
    next_function = source.index("\n}\n", start)
    return source[start:next_function]


def test_moe_gating_top_k_meta_preserves_symbolic_shapes():
    source = META_BINDING.read_text(encoding="utf-8")
    body = _function_body(source, "moe_gating_top_k_meta")

    assert "x.sym_sizes()" in body
    assert "bias.sym_sizes()" in body
    assert "at::empty_symint(topk_shape" in body
    assert "at::empty_symint(score_shape" in body
    assert "x.sizes()" not in body
    assert "at::empty({rows" not in body
