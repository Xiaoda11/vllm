import json
from pathlib import Path

import pytest

from scripts.lab_v026_workload import (
    load_scenario,
    prepare_requests,
    scale_scenario,
)

CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "scheduler_trace" / "configs"
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIRECTORY.glob("*.json")))
def test_fixed_scenarios_are_valid(config_path: Path) -> None:
    scenario = load_scenario(config_path)

    assert scenario.scenario_id == config_path.name[:2].upper()
    assert scenario.requests


def test_prompt_scale_preserves_s3_one_to_two_ratio() -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "s3_dual_simultaneous.json")

    scaled = scale_scenario(scenario, 0.5)

    assert [request.prompt_tokens for request in scaled.requests] == [4096, 8192]


def test_prepared_prompts_have_exact_lengths_and_shared_prefix() -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "s6_shared_prefix.json")

    prepared = prepare_requests(scenario, scenario, FakeTokenizer())

    assert [len(request.prompt_token_ids) for request in prepared] == [8192, 8192]
    assert prepared[0].shared_prefix_sha256
    assert prepared[0].shared_prefix_sha256 == prepared[1].shared_prefix_sha256
    shared_tokens = prepared[0].spec.shared_prefix_tokens
    assert (
        prepared[0].prompt_token_ids[:shared_tokens]
        == prepared[1].prompt_token_ids[:shared_tokens]
    )
    assert (
        prepared[0].prompt_token_ids[shared_tokens:]
        != prepared[1].prompt_token_ids[shared_tokens:]
    )


def test_duplicate_request_ids_are_rejected(tmp_path: Path) -> None:
    config = json.loads((CONFIG_DIRECTORY / "s3_dual_simultaneous.json").read_text())
    config["requests"][1]["request_id"] = "A"
    config_path = tmp_path / "duplicate.json"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_scenario(config_path)
