from pathlib import Path

from tools.probe_aclgraph_kv_connector_hooks import audit_sources, classify_replay

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_static_audit_records_connector_graph_boundary():
    audit = audit_sources(REPO_ROOT)

    if audit["upstream_mixin_available"]:
        assert audit["upstream_mixin_calls_start_load"] is True
    assert audit["connector_context_wraps_model_forward"] is True
    assert audit["graph_capture_calls_python_runnable"] is True
    assert audit["graph_replay_calls_python_runnable"] is False
    assert audit["graph_replay_uses_device_replay"] is True
    assert audit["attention_python_hook_exists"] is True
    assert audit["connector_wait_host_synchronizes_load_stream"] is True
    assert audit["connector_wait_launches_next_layer"] is True
    assert audit["cudagraph_dispatch_has_kv_connector_guard"] is False
    assert audit["cudagraph_call_only_passes_model_enforce_eager"] is True
    assert audit["mla_wait_is_prefill_conditional"] is True


def test_classify_replay_detects_missing_python_pipeline():
    finding = classify_replay(
        layers=4,
        capture_wait_calls=4,
        replay_wait_calls=4,
        replay_matches_all_new_sources=False,
    )

    assert finding == {
        "layers": 4,
        "python_hooks_rerun": False,
        "replay_matches_all_new_sources": False,
        "outcome": "python_hooks_skipped_and_layerwise_load_pipeline_was_not_replayed",
    }


def test_classify_replay_allows_captured_device_load_tasks():
    finding = classify_replay(
        layers=4,
        capture_wait_calls=4,
        replay_wait_calls=4,
        replay_matches_all_new_sources=True,
    )

    assert finding["python_hooks_rerun"] is False
    assert finding["outcome"] == "python_hooks_skipped_but_load_tasks_were_captured"
