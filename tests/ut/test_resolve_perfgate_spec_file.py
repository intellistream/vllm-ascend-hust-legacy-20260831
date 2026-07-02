# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/scripts/resolve_perfgate_spec_file.py"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location(
        "resolve_perfgate_spec_file",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_chip_model_from_npu_smi_text() -> None:
    resolver = load_resolver()

    assert resolver.detect_chip_model_from_text("Name: Ascend 910B2") == "910B2"
    assert resolver.detect_chip_model_from_text("chip model: 910B3") == "910B3"


def test_resolve_spec_file_maps_random_online_for_b2() -> None:
    resolver = load_resolver()

    spec_file, chip_model = resolver.resolve_spec_file(
        explicit_spec_file="",
        explicit_chip_model="910B2",
        scenario="random-online",
        npu_smi_bin="",
    )

    assert spec_file == "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json"
    assert chip_model == "910B2"


def test_resolve_spec_file_maps_sharegpt_online_for_b2() -> None:
    resolver = load_resolver()

    spec_file, chip_model = resolver.resolve_spec_file(
        explicit_spec_file="",
        explicit_chip_model="910B2",
        scenario="sharegpt-online",
        npu_smi_bin="",
    )

    assert (
        spec_file
        == "docs/official-baselines/"
        "perfgate-ascend-qwen25-3b-sharegpt-online-910b2.json"
    )
    assert chip_model == "910B2"


def test_resolve_spec_file_fails_closed_for_unknown_scenario() -> None:
    resolver = load_resolver()

    with pytest.raises(ValueError, match="prefix-repetition-online"):
        resolver.resolve_spec_file(
            explicit_spec_file="",
            explicit_chip_model="910B2",
            scenario="prefix-repetition-online",
            npu_smi_bin="",
        )


def test_main_writes_absolute_same_spec_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = load_resolver()
    spec_file = (
        tmp_path
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-qwen25-3b-sharegpt-online-910b2.json"
    )
    spec_file.parent.mkdir(parents=True)
    spec_file.write_text("{}", encoding="utf-8")
    github_env = tmp_path / "github-env"

    monkeypatch.setattr(
        "sys.argv",
        [
            "resolve_perfgate_spec_file.py",
            "--explicit-chip-model",
            "910B2",
            "--scenario",
            "sharegpt-online",
            "--benchmark-repo",
            str(tmp_path),
            "--github-env",
            str(github_env),
        ],
    )

    assert resolver.main() == 0
    env_text = github_env.read_text(encoding="utf-8")
    assert f"SAME_SPEC_SPEC_FILE={spec_file}" in env_text
