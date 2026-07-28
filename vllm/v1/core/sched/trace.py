# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_TRACE_PATH_ENV = "LAB_V026_SCHEDULER_TRACE_PATH"
_CLOSE_SENTINEL = object()


class JsonlTraceWriter:
    """Write trace records on a background thread."""

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
        self._queue.put(record)

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


def create_scheduler_trace_writer() -> JsonlTraceWriter | None:
    trace_path = os.getenv(SCHEDULER_TRACE_PATH_ENV)
    if not trace_path:
        return None
    writer = JsonlTraceWriter(Path(trace_path))
    logger.info("Scheduler trace enabled: %s", writer.path)
    return writer


def create_model_runner_trace_writer() -> JsonlTraceWriter | None:
    trace_path = os.getenv(SCHEDULER_TRACE_PATH_ENV)
    if not trace_path:
        return None
    scheduler_path = Path(trace_path)
    runner_path = scheduler_path.with_name(
        f"{scheduler_path.stem}.mrv2{scheduler_path.suffix}"
    )
    writer = JsonlTraceWriter(runner_path)
    logger.info("MRV2 trace enabled: %s", writer.path)
    return writer
