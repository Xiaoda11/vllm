# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_trace_module() -> ModuleType:
    trace_path = (
        Path(__file__).parents[1] / "vllm" / "v1" / "core" / "sched" / "trace.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scheduler_trace_lab_trace", trace_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace = _load_trace_module()
JsonlTraceWriter = trace.JsonlTraceWriter
TRACE_SCHEMA_VERSION = trace.TRACE_SCHEMA_VERSION


def _disable_trace(monkeypatch) -> None:
    monkeypatch.delenv(trace.SCHEDULER_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(trace.LEGACY_SCHEDULER_TRACE_PATH_ENV, raising=False)


def test_scheduler_trace_writer_is_disabled_by_default(monkeypatch) -> None:
    _disable_trace(monkeypatch)
    assert trace.create_scheduler_trace_writer() is None


def test_model_runner_trace_writer_is_disabled_by_default(monkeypatch) -> None:
    _disable_trace(monkeypatch)
    assert trace.create_model_runner_trace_writer() is None


def test_model_runner_trace_writer_uses_process_local_path(
    tmp_path: Path, monkeypatch
) -> None:
    _disable_trace(monkeypatch)
    scheduler_path = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv(trace.SCHEDULER_TRACE_PATH_ENV, str(scheduler_path))
    monkeypatch.setattr(trace.os, "getpid", lambda: 4321)

    writer = trace.create_model_runner_trace_writer()
    assert writer is not None
    try:
        assert writer.path == tmp_path / "scheduler.mrv2.pid4321.jsonl"
        writer.record({"event": "model_runner_batch", "step_id": 1})
    finally:
        writer.close()

    record = json.loads(writer.path.read_text(encoding="utf-8"))
    assert record["event"] == "model_runner_batch"
    assert record["schema_version"] == TRACE_SCHEMA_VERSION


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
