# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
from pathlib import Path

import pytest

from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.v1.core.sched.trace import (
    LEGACY_SCHEDULER_TRACE_PATH_ENV,
    SCHEDULER_TRACE_PATH_ENV,
)
from vllm.v1.executor.multiproc_executor import WorkerProc

from .utils import create_requests, create_scheduler

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _TraceEchoWorker:
    def execute_model(self, scheduler_output):
        return (
            getattr(scheduler_output, "scheduler_trace_step_id", None),
            scheduler_output.num_scheduled_tokens,
        )


class _TraceDispatchWorkerProc(WorkerProc):
    """Minimal shell that reuses WorkerProc RPC dispatch without worker init."""

    def handle_output(self, output):
        self._trace_result_conn.send(output)


def _worker_rpc_trace_reader(handle: Handle, result_conn) -> None:
    reader = MessageQueue.create_from_handle(handle, rank=0)
    worker_proc = object.__new__(_TraceDispatchWorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = _TraceEchoWorker()
    worker_proc._trace_result_conn = result_conn
    try:
        reader.wait_until_ready()
        rpc_request = reader.dequeue(timeout=10)
        worker_proc._execute_worker_rpc(rpc_request)
    finally:
        reader.shutdown()
        result_conn.close()


def _disable_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SCHEDULER_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(LEGACY_SCHEDULER_TRACE_PATH_ENV, raising=False)


def test_scheduler_trace_step_id_survives_worker_rpc_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross SHM/ZMQ IPC and execute via the real WorkerProc RPC dispatcher."""
    _disable_trace(monkeypatch)
    monkeypatch.setenv(SCHEDULER_TRACE_PATH_ENV, str(tmp_path / "worker-rpc.jsonl"))

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
            req_ids=["worker-rpc"],
        )
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        assert scheduler_output.scheduler_trace_step_id == 1
    finally:
        scheduler.shutdown()

    writer = MessageQueue(n_reader=1, n_local_reader=1)
    ctx = mp.get_context("spawn")
    result_parent, result_child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker_rpc_trace_reader,
        args=(writer.export_handle(), result_child),
    )
    proc.start()
    result_child.close()
    try:
        writer.wait_until_ready()
        # Match MultiprocExecutor.collective_rpc()'s actual wire tuple.
        writer.enqueue(("execute_model", (scheduler_output,), {}, 0))
        assert result_parent.poll(15), "WorkerProc RPC dispatch did not return in time"
        step_id, num_scheduled_tokens = result_parent.recv()
        assert step_id == 1
        assert num_scheduled_tokens == {"worker-rpc": 24}
    finally:
        writer.shutdown()
        result_parent.close()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()
            proc.join()

    assert proc.exitcode == 0
