# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_event_module() -> ModuleType:
    path = Path(__file__).parents[1] / "vllm" / "v1" / "core" / "sched" / "trace_event.py"
    spec = importlib.util.spec_from_file_location("scheduler_trace_lab_event", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


event_mod = _load_event_module()
make_scheduler_step_event = event_mod.make_scheduler_step_event


def _state(status: str, computed: int, blocks: list[list[int]]) -> dict:
    return {
        "status": status,
        "num_prompt_tokens": 4096,
        "num_output_tokens": 0,
        "num_computed_tokens": computed,
        "num_in_flight_tokens": 0,
        "num_processed_tokens": computed,
        "is_prefill_chunk": True,
        "block_ids": blocks,
    }


def test_event_explains_v028_dual_budget_and_kv_growth() -> None:
    before = {
        "running": ["A"],
        "waiting": ["B"],
        "skipped_waiting": [],
        "kv_cache_usage": 0.25,
        "num_free_blocks": 12,
        "requests": {
            "A": _state("RUNNING", 1024, [[1, 2]]),
            "B": _state("WAITING", 0, []),
        },
    }
    after = {
        "running": ["A", "B"],
        "waiting": [],
        "skipped_waiting": [],
        "kv_cache_usage": 0.5,
        "num_free_blocks": 8,
        "requests": {
            "A": _state("RUNNING", 1024, [[1, 2, 3]]),
            "B": _state("RUNNING", 0, [[4, 5, 6]]),
        },
    }

    event = make_scheduler_step_event(
        step_id=7,
        before=before,
        after=after,
        num_scheduled_tokens={"A": 512, "B": 1024},
        total_num_scheduled_tokens=1536,
        finished_req_ids=set(),
        preempted_req_ids=None,
        token_budget_initial=2048,
        token_budget_remaining=512,
        input_budget_initial=1792,
        input_budget_remaining=256,
        block_size=16,
        prefix_cached_tokens={"B": 256},
        timestamp_ns=123,
    )

    assert event["schema_version"] == 1
    assert event["step_id"] == 7
    assert event["token_budget"] == {
        "initial": 2048,
        "scheduled": 1536,
        "remaining": 512,
    }
    assert event["input_budget"] == {"initial": 1792, "remaining": 256}
    assert event["scheduled_request_ids"] == ["A", "B"]

    requests = {item["request_id"]: item for item in event["requests"]}
    assert requests["A"]["allocated_block_ids"] == [[3]]
    assert requests["B"]["allocated_block_ids"] == [[4, 5, 6]]
    assert requests["B"]["prefix_cached_tokens"] == 256


def test_event_records_preemption_and_allocation_failure() -> None:
    before = {
        "running": ["A"],
        "waiting": [],
        "skipped_waiting": [],
        "kv_cache_usage": 1.0,
        "num_free_blocks": 0,
        "requests": {"A": _state("RUNNING", 2048, [[1, 2, 3]])},
    }
    after = {
        "running": [],
        "waiting": ["A"],
        "skipped_waiting": [],
        "kv_cache_usage": 0.0,
        "num_free_blocks": 3,
        "requests": {"A": _state("PREEMPTED", 0, [])},
    }
    failure = {
        "phase": "running",
        "request_id": "A",
        "num_new_tokens": 512,
        "num_computed_tokens": 2048,
        "num_free_blocks": 0,
        "requires_kv_delivery": True,
    }

    event = make_scheduler_step_event(
        step_id=8,
        before=before,
        after=after,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        finished_req_ids=set(),
        preempted_req_ids={"A"},
        token_budget_initial=2048,
        token_budget_remaining=2048,
        input_budget_initial=2048,
        input_budget_remaining=2048,
        block_size=16,
        allocation_failures=[failure],
        timestamp_ns=456,
    )

    assert event["preempted_request_ids"] == ["A"]
    assert event["allocation_failures"] == [failure]
    request = event["requests"][0]
    assert request["freed_block_ids"] == [[1, 2, 3]]
    assert request["after"]["status"] == "PREEMPTED"
