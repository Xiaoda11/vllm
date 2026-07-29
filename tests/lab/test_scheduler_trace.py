import json
from pathlib import Path

import pytest

from scripts.lab_scheduler_trace_to_csv import flatten_trace
from vllm.v1.core.sched.trace import (
    SCHEDULER_TRACE_PATH_ENV,
    JsonlTraceWriter,
    create_model_runner_trace_writer,
    create_scheduler_trace_writer,
)


def test_trace_is_disabled_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SCHEDULER_TRACE_PATH_ENV, raising=False)

    assert create_scheduler_trace_writer() is None
    assert create_model_runner_trace_writer() is None


def test_writer_flushes_jsonl_on_close(tmp_path: Path) -> None:
    path = tmp_path / "scheduler_trace.jsonl"
    writer = JsonlTraceWriter(path)

    writer.record({"schema_version": 1, "step_id": 1})
    writer.record({"schema_version": 1, "step_id": 2})
    writer.close()
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step_id"] for record in records] == [1, 2]


def test_writers_use_separate_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler_path = tmp_path / "scheduler_trace.jsonl"
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(scheduler_path))

    scheduler_writer = create_scheduler_trace_writer()
    runner_writer = create_model_runner_trace_writer()
    assert scheduler_writer is not None
    assert runner_writer is not None
    scheduler_writer.close()
    runner_writer.close()

    assert scheduler_path.exists()
    assert (tmp_path / "scheduler_trace.mrv2.jsonl").exists()


def test_converter_joins_scheduler_and_mrv2_rows() -> None:
    scheduler_step = {
        "schema_version": 1,
        "event": "scheduler_step",
        "step_id": 1,
        "scheduled_request_ids": ["A"],
        "queues": {
            "running_before": [],
            "waiting_before": ["A"],
            "running_after": ["A"],
            "waiting_after": [],
        },
        "token_budget": {"initial": 4, "scheduled": 4, "remaining": 0},
        "kv_cache": {
            "block_size": 4,
            "usage_before": 0.0,
            "usage_after": 0.1,
            "num_free_blocks_before": 10,
            "num_free_blocks_after": 9,
        },
        "preempted_request_ids": [],
        "finished_request_ids": [],
        "requests": [
            {
                "request_id": "A",
                "before": {
                    "status": "WAITING",
                    "num_prompt_tokens": 8,
                    "num_output_tokens": 0,
                    "num_computed_tokens": 0,
                    "num_in_flight_tokens": 0,
                    "num_processed_tokens": 0,
                    "block_ids": [],
                },
                "after": {
                    "status": "RUNNING",
                    "num_prompt_tokens": 8,
                    "num_output_tokens": 0,
                    "num_computed_tokens": 4,
                    "num_in_flight_tokens": 4,
                    "num_processed_tokens": 0,
                    "block_ids": [[9]],
                },
                "num_scheduled_tokens": 4,
                "prefix_cached_tokens": 0,
                "allocated_block_ids": [[9]],
                "freed_block_ids": [[]],
            }
        ],
    }
    runner_steps = {
        1: {
            "request_ids": ["A"],
            "request_id_to_persistent_row": {"A": 15},
        }
    }

    rows = flatten_trace([scheduler_step], runner_steps)

    assert rows[0]["computed_tokens_after_schedule"] == 4
    assert rows[0]["processed_tokens_after_schedule"] == 0
    assert rows[0]["persistent_row"] == 15
    assert rows[0]["mrv2_execution_order"] == "A"
    assert rows[0]["kv_block_size"] == 4
    assert rows[0]["block_table_blocks_before"] == 0
    assert rows[0]["block_table_blocks_after"] == 1
    assert rows[0]["allocated_blocks"] == 1
    assert rows[0]["freed_blocks"] == 0
