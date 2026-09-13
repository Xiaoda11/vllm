# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pickle
from pathlib import Path
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.trace import (
    LEGACY_SCHEDULER_TRACE_PATH_ENV,
    SCHEDULER_TRACE_PATH_ENV,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _disable_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SCHEDULER_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(LEGACY_SCHEDULER_TRACE_PATH_ENV, raising=False)


def _model_output(req_id: str, token_id: int) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token_id]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


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
        assert not hasattr(output, "scheduler_trace_step_id")
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
        assert output.scheduler_trace_step_id == 1
        restored_output = pickle.loads(pickle.dumps(output))
        assert restored_output.scheduler_trace_step_id == 1
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


def test_real_scheduler_trace_records_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finish through update_from_output and trace the next-step finished flush."""
    _disable_trace(monkeypatch)
    trace_path = tmp_path / "completion.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(trace_path))

    scheduler = create_scheduler(
        max_num_batched_tokens=32,
        num_blocks=8,
        block_size=16,
    )
    try:
        (request,) = create_requests(
            num_requests=1,
            num_tokens=8,
            max_tokens=1,
            block_size=16,
            req_ids=["complete"],
        )
        scheduler.add_request(request)
        first_output = scheduler.schedule()
        scheduler.update_from_output(first_output, _model_output("complete", 101))

        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert "complete" in scheduler.finished_req_ids
        second_output = scheduler.schedule()
        assert second_output.finished_req_ids == {"complete"}
        assert second_output.num_scheduled_tokens == {}
    finally:
        scheduler.shutdown()

    records = _read_trace(trace_path)
    assert len(records) == 2
    finish_event = records[1]
    assert finish_event["step_id"] == 2
    assert finish_event["finished_request_ids"] == ["complete"]
    assert finish_event["scheduled_request_ids"] == []
    assert finish_event["queues"]["running_before"] == []
    assert finish_event["queues"]["running_after"] == []
    request_event = next(
        item for item in finish_event["requests"] if item["request_id"] == "complete"
    )
    assert request_event["before"] is None
    assert request_event["after"] is None


def test_real_scheduler_trace_records_preemption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force real KV-pressure preemption through Scheduler.schedule()."""
    _disable_trace(monkeypatch)
    trace_path = tmp_path / "preemption.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(trace_path))

    # Block 0 is reserved as the null block, so 11 total blocks means 10 usable.
    # Each 80-token prompt consumes five 16-token blocks. Both can be in flight,
    # but scheduling one more token for the first request requires preempting the
    # second request once the cache is full.
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    req0, req1 = create_requests(
        num_requests=2,
        num_tokens=80,
        block_size=16,
        req_ids=["keep-running", "preempt-me"],
    )

    try:
        scheduler.add_request(req0)
        output0 = scheduler.schedule()
        assert output0.num_scheduled_tokens == {"keep-running": 80}

        scheduler.add_request(req1)
        output1 = scheduler.schedule()
        assert output1.num_scheduled_tokens == {"preempt-me": 80}

        scheduler.update_from_output(output0, _model_output("keep-running", 7))

        output2 = scheduler.schedule()
        assert output2.num_scheduled_tokens == {"keep-running": 1}
        assert output2.preempted_req_ids == {"preempt-me"}
        assert req1.status == RequestStatus.PREEMPTED
        assert len(scheduler.running) == 1
        assert scheduler.running[0] == req0

        # Match the real overlapping-batch behavior: output from the preempted
        # in-flight request can still arrive and must be consumed safely.
        scheduler.update_from_output(output1, _model_output("preempt-me", 42))
        assert list(req1.output_token_ids) == [42]
    finally:
        scheduler.shutdown()

    records = _read_trace(trace_path)
    assert len(records) == 3
    preempt_event = records[2]
    assert preempt_event["step_id"] == 3
    assert preempt_event["preempted_request_ids"] == ["preempt-me"]
    assert preempt_event["scheduled_request_ids"] == ["keep-running"]
    assert preempt_event["queues"]["running_before"] == [
        "keep-running",
        "preempt-me",
    ]
    assert preempt_event["queues"]["running_after"] == ["keep-running"]
    assert "preempt-me" in preempt_event["queues"]["waiting_after"]

    request_event = next(
        item for item in preempt_event["requests"] if item["request_id"] == "preempt-me"
    )
    assert request_event["before"]["status"] == "RUNNING"
    assert request_event["after"]["status"] == "PREEMPTED"
    assert any(request_event["freed_block_ids"])


def test_real_scheduler_trace_records_sync_kv_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Propagate a synchronous connector load into a real scheduler trace."""
    _disable_trace(monkeypatch)
    trace_path = tmp_path / "sync-kv.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(trace_path))
    (tmp_path / "config.json").write_text(
        '{"architectures": ["OPTForCausalLM"], "model_type": "opt"}',
        encoding="utf-8",
    )

    block_size = 16
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=block_size, is_async=False),
        block_size=block_size,
    )
    try:
        (request,) = create_requests(
            num_requests=1,
            num_tokens=block_size * 2,
            block_size=block_size,
            req_ids=["sync-kv"],
        )
        scheduler.add_request(request)
        assert scheduler.connector is not None
        scheduler.connector.get_num_new_matched_tokens = Mock(
            return_value=(block_size, False)
        )

        output = scheduler.schedule()
        assert output.has_sync_kv_loads is True
        assert output.num_scheduled_tokens == {"sync-kv": block_size}
    finally:
        scheduler.shutdown()

    (event,) = _read_trace(trace_path)
    assert event["scheduled_request_ids"] == ["sync-kv"]
    assert event["token_budget"]["scheduled"] == block_size
    assert event["kv_connector"]["has_sync_kv_loads"] is True
