# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_build_requirements_only_contain_setup_dependencies():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    build_requirements = tomllib.loads(pyproject.read_text())["build-system"]["requires"]

    package_names = {
        requirement.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].lower() for requirement in build_requirements
    }
    runtime_only_packages = {
        "fastapi",
        "torch",
        "torch-npu",
        "transformers",
        "triton-ascend",
    }

    assert package_names.isdisjoint(runtime_only_packages)
