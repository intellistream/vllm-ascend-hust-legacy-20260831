#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "_selected_tests.yaml"
SMART_UT_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr_smart_ut.yaml"
PR_TEST_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pr_test.yaml"
RUNNER_LABEL_PATH = REPO_ROOT / ".github" / "workflows" / "scripts" / "runner_label.json"
RUN_SELECTED_TESTS_PATH = REPO_ROOT / ".github" / "workflows" / "scripts" / "run_selected_tests.sh"
REJECTION_SAMPLER_UTILS_PATH = (
    REPO_ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "rejection_sampler_utils.py"
)
BALANCE_SCHEDULER_PATH = REPO_ROOT / "vllm_ascend" / "patch" / "platform" / "patch_balance_schedule.py"


def test_a2_single_npu_container_uses_runner_scoped_runtime_contract() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    runner_labels = RUNNER_LABEL_PATH.read_text(encoding="utf-8")

    assert '"linux-aarch64-a2b3-1"' in runner_labels
    assert "matrix.group.runner == 'linux-aarch64-a2b3-1'" in workflow
    assert "--device /dev/davinci1" in workflow
    assert "--device /dev/davinci_manager" in workflow
    assert "--device /dev/devmm_svm" in workflow
    assert "--device /dev/hisi_hdc" in workflow
    assert "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" in workflow
    assert "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro" in workflow
    assert "matrix.group.device_runner == true && '0'" in workflow
    assert "matrix.group.device_runner == true" in workflow
    assert "/dev/davinci{0}:/dev/davinci0" in workflow
    assert "/data/shared_models/modelscope_cache:/__modelscope_seed:ro" in workflow
    assert "/data/actions-runners/huggingface-cache-npu{0}:/github/home/.cache/huggingface" in workflow
    assert "/data/actions-runners/vllm-assets:/github/home/.cache/vllm/assets:ro" in workflow
    assert 'cp -as /__modelscope_seed/. "${modelscope_cache}/models/"' in workflow
    assert r"\( -name .mdl -o -name .msc -o -name .mv \)" in workflow
    assert 'echo "MODELSCOPE_CACHE=${modelscope_cache}" >> "$GITHUB_ENV"' in workflow
    assert 'test ! -L "${modelscope_cache}/models/Qwen/Qwen3-0___6B/.mdl"' in workflow
    assert 'if [ -z "${MODELSCOPE_CACHE:-}" ]; then' in workflow
    assert 'rm -rf --one-file-system "$MODELSCOPE_CACHE"' in workflow


def test_npu_preflight_is_fail_closed_and_runs_before_package_install() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    preflight = "check_ascend_container_runtime.py"

    assert preflight in workflow
    assert workflow.index(preflight) < workflow.index("- name: Install packages")
    preflight_block = workflow[
        workflow.index("- name: Validate Ascend container runtime") : workflow.index("- name: Install packages")
    ]
    assert "continue-on-error" not in preflight_block
    assert "|| true" not in preflight_block
    assert "if: ${{ matrix.group.npu_type != 'cpu' }}" in preflight_block


def test_device_checkout_prefers_the_runner_local_git_mirrors() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    selected_job = workflow[workflow.index("  selected-tests:") :]

    assert "cache=/__git-cache/vllm-ascend-hust.git" in workflow
    assert "cache=/__git-cache/vllm.git" in workflow
    assert workflow.count('git --git-dir="$cache" cat-file -e') == 3
    assert 'cat-file -e "${EXPECTED_REF_SHA}^{commit}"' in workflow
    assert '[[ "$checkout_ref" =~ ^[0-9a-f]{40}$ ]]' in workflow
    assert workflow.count("git remote remove origin 2>/dev/null || true") == 5
    assert workflow.count("git fetch --no-tags --filter=blob:none --depth=1") == 2
    assert "uses: actions/checkout@" not in selected_job


def test_tag_checkout_exports_the_upstream_compatibility_version() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    checkout_start = workflow.index("- name: Checkout vllm-project/vllm repo")
    install_start = workflow.index("- name: Install vllm-project/vllm from source")
    checkout_block = workflow[checkout_start:install_start]
    assert '[[ "$VLLM_REF" =~ ^v(' in checkout_block
    assert 'echo "VLLM_VERSION=${BASH_REMATCH[1]}" >> "$GITHUB_ENV"' in checkout_block


def test_standalone_a2_runner_does_not_depend_on_cluster_local_package_cache() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        "(matrix.group.device_runner == true || matrix.group.runner == 'linux-aarch64-a2b3-1') && 'https://pypi.org/simple'"
    ) in workflow
    selected_job = workflow[workflow.index("  selected-tests:") :]
    container_env = selected_job[selected_job.index("      env:") : selected_job.index("    steps:")]
    standalone_extra_index = (
        "(matrix.group.device_runner == true || matrix.group.runner == 'linux-aarch64-a2b3-1') && "
        "'https://repo.huaweicloud.com/ascend/repos/pypi'"
    )
    assert standalone_extra_index in container_env
    assert "https://download.pytorch.org/whl/cpu/" not in container_env
    install_block = workflow[
        workflow.index("- name: Install packages") : workflow.index("- name: Checkout vllm-project/vllm repo")
    ]
    assert 'if [ "${{ matrix.group.device_runner }}" = "true" ]' in install_block
    assert 'command -v "$tool"' in install_block
    assert "exit 0" in install_block
    assert '[ "${{ matrix.group.runner }}" != "linux-aarch64-a2b3-1" ]' in install_block
    assert "cache-service.nginx-pypi-cache.svc.cluster.local:8081" in install_block


def test_device_runners_restore_and_save_csrc_cache_on_cache_miss() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "/data/actions-runners/csrc-cache-artifacts:/__csrc-cache:ro" in workflow
    assert "/data/actions-runners/tools/zstd:/usr/local/bin/zstdmt:ro" in workflow
    assert "git gcc g++ cmake curl uv zstd zstdmt" in workflow
    restore_start = workflow.index("- name: Restore runner-local vllm-ascend csrc cache")
    install_start = workflow.index("- name: Install vllm-project/vllm-ascend with device")
    restore_block = workflow[restore_start:install_start]
    assert "/__csrc-cache/${{ steps.get_csrc_hash.outputs.CSRC_HASH }}/vllm_ascend" in restore_block
    assert 'cp -a "${local_cache}/." vllm_ascend/' in restore_block
    assert "matrix.group.device_runner == true && steps.cache-csrc.outputs.cache-hit != 'true'" in restore_block

    save_start = workflow.index("- name: Save vllm-ascend csrc cache")
    verify_start = workflow.index("- name: Verify required AscendC custom ops are registered (310p)")
    save_block = workflow[save_start:verify_start]
    assert "steps.cache-csrc.outputs.cache-hit != 'true'" in save_block
    assert "steps.csrc-filter.outputs.csrc == 'true'" not in save_block


def test_spec_decode_imports_helpers_exposed_by_both_pinned_vllm_revisions() -> None:
    source = REJECTION_SAMPLER_UTILS_PATH.read_text(encoding="utf-8")
    import_block = source[source.index("from vllm.v1.worker.gpu.spec_decode") : source.index("@triton.jit")]

    assert "_compute_block_stats_kernel," in import_block
    assert "_compute_global_lse," in import_block
    assert "_compute_global_logsumexp" not in import_block
    assert "_compute_local_logits_stats_kernel" not in import_block


def test_disabled_balance_scheduler_preserves_the_upstream_schedule_signature() -> None:
    source = BALANCE_SCHEDULER_PATH.read_text(encoding="utf-8")
    fallback = source[source.index("def schedule(") : source.index("# NOTE(woosuk)")]

    assert "return super().schedule()" in fallback
    assert "return super().schedule(throttle_prefills)" not in fallback


def test_device_runners_materialize_torch_npu_cache_links() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    start = workflow.index("- name: Materialize torch-npu paths rejected by symlink checks")
    end = workflow.index("- name: Save vllm-ascend csrc cache")
    block = workflow[start:end]
    assert "matrix.group.device_runner == true" in block
    assert 'root = Path(site.getsitepackages()[0]) / "torch_npu"' in block
    assert "path.resolve(strict=True)" in block
    assert "shutil.copy2(target, materialized, follow_symlinks=True)" in block
    assert 'test ! -L "${torch_npu_root}/lib/libop_plugin_atb.so"' in block


def test_failure_summary_preserves_numeric_pytest_exit_status() -> None:
    script = RUN_SELECTED_TESTS_PATH.read_text(encoding="utf-8")

    summary = script[script.index("print_summary()") : script.index("run_pytest_target()")]
    assert "local result target result_status log_file" in summary
    assert "read -r target result_status log_file" in summary
    assert "read -r target status log_file" not in summary
    assert 'exit "${status}"' in script


def test_selected_device_tests_retry_only_transient_npu_memory_contention() -> None:
    script = RUN_SELECTED_TESTS_PATH.read_text(encoding="utf-8")

    assert 'npu_resource_retry_attempts="${NPU_RESOURCE_RETRY_ATTEMPTS:-3}"' in script
    assert 'npu_resource_retry_delay_seconds="${NPU_RESOURCE_RETRY_DELAY_SECONDS:-30}"' in script
    assert 'attempt_log="${log_file}.attempt"' in script
    assert "Free memory on device .* is less than desired GPU memory utilization" in script
    assert '"${attempt_log}"' in script
    assert '"${mode}" != "with-device"' in script
    assert "Transient NPU memory contention detected; keeping the runner busy" in script


def test_selected_test_artifacts_are_unique_and_scratch_is_released() -> None:
    script = RUN_SELECTED_TESTS_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'selected_test_log_scope="${SELECTED_TEST_LOG_SCOPE:-local}"' in script
    assert "selected-tests-${npu_type}-${num_npus}card-${selected_test_log_scope}" in script
    assert "SELECTED_TEST_LOG_DIR" in script
    assert "SELECTED_TEST_VLLM_CACHE_ROOT" in script
    assert "SELECTED_TEST_LOG_SCOPE:" in workflow
    assert "selected-test-logs-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert "timing-data-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert "${{ matrix.group.partition }}" in workflow
    assert "Remove per-job selected-test scratch" in workflow
    assert 'rm -rf --one-file-system "$SELECTED_TEST_LOG_DIR"' in workflow
    assert 'rm -rf --one-file-system "$SELECTED_TEST_VLLM_CACHE_ROOT"' in workflow


def test_selected_tests_deliver_exact_merge_source_as_hosted_bundle() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "prepare-ascend-source:" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "fetch-depth: 0" in workflow
    assert 'git bundle create "${RUNNER_TEMP}/vllm-ascend-source.bundle" codex-exact-source' in workflow
    assert "needs: prepare-ascend-source" in workflow
    assert "Download exact Ascend source bundle" in workflow
    assert 'exact_bundle="${RUNNER_TEMP}/exact-ascend-source/vllm-ascend-source.bundle"' in workflow
    assert 'git remote add origin "$exact_bundle"' in workflow
    assert "git fetch --no-tags origin codex-exact-source" in workflow
    assert 'if [ -n "$EXPECTED_REF_SHA" ] && [ -f "$exact_bundle" ]; then' in workflow


def test_smart_ut_uses_the_verified_vllm_main_commit() -> None:
    workflow = SMART_UT_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert ".github/vllm-main-verified.commit" in workflow
    assert "mapfile -t verified_lines" in workflow
    assert "${#verified_lines[@]} != 1" in workflow
    assert '[[ ! "${verified_lines[0]}" =~ ^[0-9a-f]{40}$ ]]' in workflow
    assert "tr -d '[:space:]'" not in workflow
    assert "main_commit: ${{ steps.vllm.outputs.main_commit }}" in workflow
    assert "vllm: ${{ needs.scope.outputs.main_commit }}" in workflow
    assert "d886c26d4d4fef7d079696beb4ece1cfb4b008a8" not in workflow


def test_package_builds_do_not_auto_load_the_torch_device_backend() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    selected_job = workflow[workflow.index("  selected-tests:") :]
    container_env = selected_job[selected_job.index("      env:") : selected_job.index("    steps:")]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD: 0" in container_env
    assert "GIT_CONFIG_KEY_0: http.version" in container_env
    assert "GIT_CONFIG_VALUE_0: HTTP/1.1" in container_env


def test_pull_request_workflows_test_the_server_generated_merge_commit() -> None:
    selected_workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    pr_workflow = PR_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")
    smart_ut_workflow = SMART_UT_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "ref_is_merge_commit:" in selected_workflow
    assert "CHECKOUT_REF: ${{ inputs.ref || github.ref }}" in selected_workflow
    assert 'checkout_ref="$CHECKOUT_REF"' in selected_workflow
    assert 'checkout_ref="$EXPECTED_REF_SHA"' in selected_workflow
    assert "EXPECTED_REF_SHA: ${{ inputs.ref_is_merge_commit && github.sha || '' }}" in selected_workflow
    assert 'if [ -n "$EXPECTED_REF_SHA" ]' in selected_workflow
    assert "expected ${EXPECTED_REF_SHA}" in selected_workflow
    assert "if: ${{ !inputs.ref_is_merge_commit }}" in selected_workflow
    for workflow in (pr_workflow, smart_ut_workflow):
        assert "ref: ${{ github.ref }}" in workflow
        assert "ref_is_merge_commit: true" in workflow
        assert "ref: ${{ github.event.pull_request.head.sha }}" not in workflow


def test_labeled_e2e_retriggers_after_new_commits() -> None:
    workflow = PR_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "- synchronize" in workflow
    assert "github.event.action == 'synchronize'" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'e2e')" in workflow


def test_hust_e2e_only_emits_groups_for_available_runner_families() -> None:
    workflow = PR_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "--allowed-runner linux-aarch64-a2b3-1" in workflow
    for device_id in (0, 1, 3, 4, 5, 6):
        assert f"--runner-pool linux-aarch64-a2b3-npu{device_id}={device_id}" in workflow
    assert "--runner-pool linux-aarch64-a2b3-npu2=2" not in workflow
