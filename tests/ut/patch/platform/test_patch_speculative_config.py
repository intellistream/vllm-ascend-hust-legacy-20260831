# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform import patch_speculative_config


class _SpeculativeConfig:
    def __init__(self, draft_model_config=None, use_dspark=True, include_draft=True):
        if include_draft:
            self.draft_model_config = draft_model_config
        self._use_dspark = use_dspark

    def use_dspark(self):
        return self._use_dspark


@pytest.fixture
def stub_original_post_init(monkeypatch):
    calls = []
    monkeypatch.setattr(patch_speculative_config, "_orig_post_init", lambda self: calls.append(self))
    return calls


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_SpeculativeConfig(include_draft=False), "draft_model_config"),
        (_SpeculativeConfig(draft_model_config=SimpleNamespace()), "hf_config"),
    ],
)
def test_dspark_requires_draft_and_hf_config(config, message, stub_original_post_init):
    with pytest.raises(ValueError, match=f"DSpark.*{message}"):
        patch_speculative_config._dspark_post_init(config)

    assert stub_original_post_init == [config]


def test_dspark_requires_a_placeholder_token_id(stub_original_post_init):
    config = _SpeculativeConfig(draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()))

    with pytest.raises(
        ValueError,
        match="DSpark.*ptd_token_id.*dspark_noise_token_id.*mask_token_id",
    ):
        patch_speculative_config._dspark_post_init(config)

    assert stub_original_post_init == [config]


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (
            SimpleNamespace(ptd_token_id=11, dspark_noise_token_id=12, mask_token_id=13),
            11,
        ),
        (
            SimpleNamespace(ptd_token_id=None, dspark_noise_token_id=12, mask_token_id=13),
            12,
        ),
        (
            SimpleNamespace(ptd_token_id=None, dspark_noise_token_id=None, mask_token_id=13),
            13,
        ),
    ],
)
def test_dspark_normalizes_placeholder_token_id(hf_config, expected, stub_original_post_init):
    config = _SpeculativeConfig(draft_model_config=SimpleNamespace(hf_config=hf_config))

    patch_speculative_config._dspark_post_init(config)

    assert hf_config.ptd_token_id == expected
    assert stub_original_post_init == [config]


def test_non_dspark_config_keeps_existing_post_init_behavior(stub_original_post_init):
    config = _SpeculativeConfig(use_dspark=False, include_draft=False)

    patch_speculative_config._dspark_post_init(config)

    assert stub_original_post_init == [config]
