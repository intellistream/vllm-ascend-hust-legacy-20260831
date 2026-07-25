from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_WORKFLOW = REPO_ROOT / ".github/workflows/pr_test.yaml"
BENCHMARK_WORKFLOW = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"


def test_labeled_e2e_retriggers_after_new_commits() -> None:
    workflow = E2E_WORKFLOW.read_text(encoding="utf-8")

    assert "- synchronize" in workflow
    assert "github.event.action == 'synchronize'" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'e2e')" in workflow


def test_labeled_benchmark_retriggers_after_new_commits() -> None:
    workflow = BENCHMARK_WORKFLOW.read_text(encoding="utf-8")

    assert "types: [labeled, synchronize]" in workflow
    assert "github.event.action == 'synchronize'" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'ready')" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'verified')" in workflow
