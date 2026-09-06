# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in JSONL tracing helpers for the Scheduler Trace Lab.

This module is intentionally independent from GPU execution. It records only
objects supplied by scheduler/model-runner instrumentation and performs file IO
on a background thread.
"""

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

TRACE_SCHEMA_VERSION = 1
SCHEDULER_TRACE_PATH_ENV = "VLLM_LAB_SCHEDULER_TRACE_PATH"
LEGACY_SCHEDULER_TRACE_PATH_ENV = "LAB_V026_SCHEDULER_TRACE_PATH"
_CLOSE_SENTINEL = object()


def tensor_metadata(tensor: Any) -> dict[str, Any]:
    """Return tensor metadata without reading tensor contents."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def tensor_layout_metadata(tensor: Any) -> dict[str, Any]:
    """Return tensor shape/layout metadata without reading tensor contents."""
    return {
        **tensor_metadata(tensor),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
    }


def _with_schema_version(record: dict[str, Any]) -> dict[str, Any]:
    if "schema_version" in record:
        return record
    return {"schema_version": TRACE_SCHEMA_VERSION, **record}


class JsonlTraceWriter:
    """Write versioned trace records on a background thread."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8")
        self._queue: queue.SimpleQueue[dict[str, Any] | object] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop,
            name=f"jsonl-trace-{path.name}",
            daemon=True,
        )
        self._thread.start()

    def record(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot record to a closed trace writer")
        self._queue.put(_with_schema_version(record))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_CLOSE_SENTINEL)
        self._thread.join()

    def _write_loop(self) -> None:
        try:
            while True:
                record = self._queue.get()
                if record is _CLOSE_SENTINEL:
                    break
                self._file.write(
                    json.dumps(record, separators=(",", ":"), sort_keys=True)
                )
                self._file.write("\n")
                self._file.flush()
        finally:
            self._file.close()


def _trace_path_from_env() -> str | None:
    trace_path = os.getenv(SCHEDULER_TRACE_PATH_ENV)
    if trace_path:
        return trace_path
    legacy_path = os.getenv(LEGACY_SCHEDULER_TRACE_PATH_ENV)
    if legacy_path:
        logger.warning(
            "%s is deprecated for the v0.28 migration; use %s instead",
            LEGACY_SCHEDULER_TRACE_PATH_ENV,
            SCHEDULER_TRACE_PATH_ENV,
        )
    return legacy_path


def create_scheduler_trace_writer() -> JsonlTraceWriter | None:
    trace_path = _trace_path_from_env()
    if not trace_path:
        return None
    writer = JsonlTraceWriter(Path(trace_path))
    logger.info("Scheduler trace enabled: %s", writer.path)
    return writer


def create_model_runner_trace_writer() -> JsonlTraceWriter | None:
    trace_path = _trace_path_from_env()
    if not trace_path:
        return None
    scheduler_path = Path(trace_path)
    runner_path = scheduler_path.with_name(
        f"{scheduler_path.stem}.mrv2{scheduler_path.suffix}"
    )
    writer = JsonlTraceWriter(runner_path)
    logger.info("MRV2 trace enabled: %s", writer.path)
    return writer
