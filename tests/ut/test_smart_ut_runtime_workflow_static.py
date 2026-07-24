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
    assert "ASCEND_RT_VISIBLE_DEVICES" in workflow
    assert "matrix.group.device_runner == true" in workflow
    assert "/dev/davinci{0}" in workflow


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


def test_container_checkout_uses_runner_compatible_node_runtime() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow.count("uses: actions/checkout@v6.0.1") == 2
    assert "uses: actions/checkout@v7" not in workflow


def test_standalone_a2_runner_does_not_depend_on_cluster_local_package_cache() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        "(matrix.group.device_runner == true || matrix.group.runner == 'linux-aarch64-a2b3-1') && 'https://pypi.org/simple'"
    ) in workflow
    container_env = workflow[workflow.index("      env:") : workflow.index("    steps:")]
    standalone_extra_index = (
        "(matrix.group.device_runner == true || matrix.group.runner == 'linux-aarch64-a2b3-1') && "
        "'https://repo.huaweicloud.com/ascend/repos/pypi'"
    )
    assert standalone_extra_index in container_env
    assert "https://download.pytorch.org/whl/cpu/" not in container_env
    install_block = workflow[
        workflow.index("- name: Install packages") : workflow.index("- name: Checkout vllm-project/vllm repo")
    ]
    assert 'if [ "${{ matrix.group.device_runner }}" != "true" ]' in install_block
    assert '[ "${{ matrix.group.runner }}" != "linux-aarch64-a2b3-1" ]' in install_block
    assert "cache-service.nginx-pypi-cache.svc.cluster.local:8081" in install_block


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

    container_env = workflow[workflow.index("      env:") : workflow.index("    steps:")]
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD: 0" in container_env
    assert "GIT_CONFIG_KEY_0: http.version" in container_env
    assert "GIT_CONFIG_VALUE_0: HTTP/1.1" in container_env


def test_pull_request_workflows_test_the_server_generated_merge_commit() -> None:
    selected_workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    pr_workflow = PR_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")
    smart_ut_workflow = SMART_UT_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "ref_is_merge_commit:" in selected_workflow
    assert "fetch-depth: ${{ inputs.ref_is_merge_commit && 1 || 0 }}" in selected_workflow
    assert "if: ${{ !inputs.ref_is_merge_commit }}" in selected_workflow
    for workflow in (pr_workflow, smart_ut_workflow):
        assert "ref: ${{ github.ref }}" in workflow
        assert "ref_is_merge_commit: true" in workflow
        assert "ref: ${{ github.event.pull_request.head.sha }}" not in workflow


def test_hust_e2e_only_emits_groups_for_available_runner_families() -> None:
    workflow = PR_TEST_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "--allowed-runner linux-aarch64-a2b3-1" in workflow
    for device_id in range(7):
        assert f"--runner-pool linux-aarch64-a2b3-npu{device_id}={device_id}" in workflow
