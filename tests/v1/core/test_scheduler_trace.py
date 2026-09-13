# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.v1.core.sched.trace import (
    LEGACY_SCHEDULER_TRACE_PATH_ENV,
    SCHEDULER_TRACE_PATH_ENV,
)

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _disable_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SCHEDULER_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(LEGACY_SCHEDULER_TRACE_PATH_ENV, raising=False)


def test_real_scheduler_trace_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace-off must keep the real Scheduler on the ordinary schedule path."""
    _disable_trace(monkeypatch)
    scheduler = create_scheduler(max_num_batched_tokens=32, block_size=16)
    try:
        assert scheduler.scheduler_trace_writer is None
        (request,) = create_requests(
            num_requests=1,
            num_tokens=16,
            block_size=16,
            req_ids=["trace-off"],
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"trace-off": 16}
    finally:
        scheduler.shutdown()


def test_real_scheduler_trace_records_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise trace emission through a real Scheduler and KVCacheManager."""
    _disable_trace(monkeypatch)
    trace_path = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(trace_path))

    scheduler = create_scheduler(
        max_num_batched_tokens=32,
        num_blocks=8,
        block_size=16,
    )
    try:
        (request,) = create_requests(
            num_requests=1,
            num_tokens=24,
            block_size=16,
            req_ids=["trace-on"],
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"trace-on": 24}
    finally:
        scheduler.shutdown()

    records = _read_trace(trace_path)
    assert len(records) == 1
    event = records[0]
    assert event["schema_version"] == 1
    assert event["event"] == "scheduler_step"
    assert event["step_id"] == 1
    assert event["scheduled_request_ids"] == ["trace-on"]
    assert event["token_budget"]["scheduled"] == 24
    assert event["token_budget"]["remaining"] == 8
    assert event["input_budget"]["initial"] >= event["input_budget"]["remaining"]
    assert event["kv_connector"] == {
        "has_sync_kv_loads": False,
        "offered_block_state": None,
    }

    request_event = next(
        item for item in event["requests"] if item["request_id"] == "trace-on"
    )
    assert request_event["before"]["status"] == "WAITING"
    assert request_event["after"]["status"] == "RUNNING"
    assert request_event["num_scheduled_tokens"] == 24
    assert any(request_event["allocated_block_ids"])
    assert event["kv_cache"]["num_free_blocks_after"] < event["kv_cache"][
        "num_free_blocks_before"
    ]


def test_real_scheduler_trace_records_waiting_allocation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record a real waiting-path allocation failure without model execution."""
    _disable_trace(monkeypatch)
    trace_path = tmp_path / "allocation-failure.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(trace_path))

    # Block 0 is the null block, leaving only one allocatable block. A 32-token
    # prefill at block size 16 needs two blocks and therefore cannot be admitted.
    scheduler = create_scheduler(
        max_num_batched_tokens=32,
        num_blocks=2,
        block_size=16,
    )
    try:
        (request,) = create_requests(
            num_requests=1,
            num_tokens=32,
            block_size=16,
            req_ids=["no-space"],
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert "no-space" not in output.num_scheduled_tokens
    finally:
        scheduler.shutdown()

    (event,) = _read_trace(trace_path)
    assert event["scheduled_request_ids"] == []
    assert event["token_budget"]["scheduled"] == 0
    assert event["queues"]["waiting_before"] == ["no-space"]
    assert event["queues"]["waiting_after"] == ["no-space"]
    assert len(event["allocation_failures"]) == 1
    failure = event["allocation_failures"][0]
    assert failure["phase"] == "waiting"
    assert failure["request_id"] == "no-space"
    assert failure["num_new_tokens"] == 32
    assert failure["num_free_blocks"] == 1
