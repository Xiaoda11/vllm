import json
from pathlib import Path

import pytest

from scripts.lab_v026_workload import (
    TokenTiming,
    load_scenario,
    override_num_gpu_blocks,
    override_prefix_caching,
    override_token_budget,
    prepare_requests,
    scale_scenario,
    write_token_timing_csv,
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


def test_token_budget_override_preserves_requests() -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "s3_dual_simultaneous.json")

    overridden = override_token_budget(scenario, 2048)

    assert overridden.engine["max_num_batched_tokens"] == 2048
    assert overridden.requests == scenario.requests
    assert scenario.engine["max_num_batched_tokens"] == 4096


def test_token_budget_override_rejects_non_positive_value() -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "s3_dual_simultaneous.json")

    with pytest.raises(ValueError, match="token_budget"):
        override_token_budget(scenario, 0)


def test_day7_pressure_scenario_has_controlled_kv_capacity() -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "d7_preemption_pressure.json")

    pressured = override_num_gpu_blocks(scenario, 1450)

    assert "num_gpu_blocks_override" not in scenario.engine
    assert pressured.engine["num_gpu_blocks_override"] == 1450
    assert scenario.engine["enable_prefix_caching"] is False
    assert scenario.engine["scheduler_reserve_full_isl"] is False
    assert [request.output_tokens for request in scenario.requests] == [16, 32]


@pytest.mark.parametrize(
    ("config_name", "expected_b_prompt_tokens"),
    [
        ("s5_decode_then_prefill_8k.json", 8192),
        ("s5_decode_then_prefill.json", 16384),
    ],
)
def test_day10_scenarios_change_only_b_prompt_length(
    config_name: str, expected_b_prompt_tokens: int
) -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / config_name)

    assert [request.request_id for request in scenario.requests] == ["A", "B"]
    assert scenario.requests[0].prompt_tokens == 1024
    assert scenario.requests[0].output_tokens == 512
    assert scenario.requests[0].arrival_s == 0.0
    assert scenario.requests[1].prompt_tokens == expected_b_prompt_tokens
    assert scenario.requests[1].output_tokens == 32
    assert scenario.requests[1].arrival_s == 1.0
    assert scenario.engine["stream_interval"] == 1


@pytest.mark.parametrize(("mode", "enabled"), [("on", True), ("off", False)])
def test_prefix_caching_override_preserves_requests(
    mode: str, enabled: bool
) -> None:
    scenario = load_scenario(CONFIG_DIRECTORY / "s6_shared_prefix.json")

    overridden = override_prefix_caching(scenario, mode)

    assert overridden.engine["enable_prefix_caching"] is enabled
    assert overridden.requests == scenario.requests
    assert scenario.engine["enable_prefix_caching"] is True


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


def test_token_timing_csv_preserves_missing_and_exact_itl(tmp_path: Path) -> None:
    path = tmp_path / "token_timing.csv"
    timings = [
        TokenTiming("run", "S5", "A", 2, 1, 2, 0.3, 0.1, 0.1),
        TokenTiming("run", "S5", "A", 1, 1, 1, 0.2, None, None),
    ]

    write_token_timing_csv(path, timings)

    assert path.read_text().splitlines() == [
        "run_id,scenario_id,request_id,event_index,chunk_tokens,"
        "cumulative_output_tokens,emitted_s,inter_event_s,single_token_itl_s",
        "run,S5,A,1,1,1,0.2,,",
        "run,S5,A,2,1,2,0.3,0.1,0.1",
    ]
