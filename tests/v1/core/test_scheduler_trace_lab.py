# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from vllm.v1.core.sched.trace import JsonlTraceWriter, TRACE_SCHEMA_VERSION


def test_jsonl_trace_writer_adds_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "scheduler.jsonl"
    writer = JsonlTraceWriter(path)
    writer.record({"event": "schedule", "step_id": 7})
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "event": "schedule",
        "schema_version": TRACE_SCHEMA_VERSION,
        "step_id": 7,
    }


def test_jsonl_trace_writer_preserves_explicit_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "scheduler.jsonl"
    writer = JsonlTraceWriter(path)
    writer.record({"schema_version": 99, "event": "synthetic"})
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 99


def test_jsonl_trace_writer_rejects_record_after_close(tmp_path: Path) -> None:
    writer = JsonlTraceWriter(tmp_path / "scheduler.jsonl")
    writer.close()

    try:
        writer.record({"event": "late"})
    except RuntimeError as exc:
        assert "closed trace writer" in str(exc)
    else:
        raise AssertionError("record() should fail after close()")
