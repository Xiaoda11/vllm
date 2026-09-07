# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).parents[1]
SCHEDULER_PATH = ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
TRACE_EVENT_PATH = ROOT / "vllm" / "v1" / "core" / "sched" / "trace_event.py"


def _load_event_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "scheduler_trace_lab_event", TRACE_EVENT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler_class() -> ast.ClassDef:
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )


def _scheduler_method(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _scheduler_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(method: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _load_trace_methods() -> type:
    wanted = {
        "_make_scheduler_trace_allocation_failure",
        "_make_scheduler_trace_snapshot",
        "_make_scheduler_trace_request_state",
        "_make_scheduler_trace_event",
    }
    methods = [
        node
        for node in _scheduler_class().body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {method.name for method in methods} == wanted
    class_node = ast.ClassDef(
        name="SchedulerTraceMethods",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module_node = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module_node)
    namespace: dict[str, Any] = {
        "Any": Any,
        "make_scheduler_step_event": _load_event_module().make_scheduler_step_event,
    }
    exec(compile(module_node, SCHEDULER_PATH, "exec"), namespace)
    return namespace["SchedulerTraceMethods"]


class _BlockPool:
    def get_num_free_blocks(self) -> int:
        return 7


class _KVCacheManager:
    usage = 0.25
    block_pool = _BlockPool()

    def __init__(self) -> None:
        self.block_ids = {"A": ([1, 2],)}

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.block_ids[request_id]


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="A",
        status=SimpleNamespace(name="RUNNING"),
        num_prompt_tokens=64,
        output_token_ids=[11, 12],
        num_computed_tokens=32,
        num_in_flight_tokens=8,
        is_prefill_chunk=True,
    )


def test_scheduler_hot_path_wires_and_closes_opt_in_trace() -> None:
    assert "create_scheduler_trace_writer" in _called_names(
        _scheduler_method("__init__")
    )
    assert {
        "_make_scheduler_trace_snapshot",
        "_make_scheduler_trace_allocation_failure",
        "_make_scheduler_trace_event",
        "record",
    } <= _called_names(_scheduler_method("schedule"))
    assert "close" in _called_names(_scheduler_method("shutdown"))

    schedule_constants = {
        node.value
        for node in ast.walk(_scheduler_method("schedule"))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"running", "waiting"} <= schedule_constants

    failure_constants = {
        node.value
        for node in ast.walk(
            _scheduler_method("_make_scheduler_trace_allocation_failure")
        )
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "token_budget_remaining",
        "input_budget_remaining",
        "requires_kv_delivery",
    } <= failure_constants


def test_scheduler_trace_methods_capture_live_v028_state() -> None:
    trace_methods = _load_trace_methods()
    scheduler = trace_methods()
    request = _request()
    scheduler.current_step = 9
    scheduler.block_size = 16
    scheduler.requires_kv_delivery = True
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.kv_cache_manager = _KVCacheManager()

    before = scheduler._make_scheduler_trace_snapshot()
    assert before["requests"]["A"]["num_processed_tokens"] == 24
    assert before["requests"]["A"]["block_ids"] == [[1, 2]]

    request.num_computed_tokens = 48
    request.num_in_flight_tokens = 24
    scheduler.kv_cache_manager.block_ids["A"] = ([1, 2, 3],)
    scheduler.kv_cache_manager.usage = 0.5
    output = SimpleNamespace(
        num_scheduled_tokens={"A": 16},
        total_num_scheduled_tokens=16,
        finished_req_ids=set(),
        preempted_req_ids=None,
    )

    event = scheduler._make_scheduler_trace_event(
        before=before,
        scheduler_output=output,
        token_budget_initial=128,
        token_budget_remaining=112,
        input_budget_initial=96,
        input_budget_remaining=80,
        prefix_cached_tokens={"A": 16},
        allocation_failures=[],
    )

    assert event["step_id"] == 9
    assert event["token_budget"] == {
        "initial": 128,
        "scheduled": 16,
        "remaining": 112,
    }
    assert event["input_budget"] == {"initial": 96, "remaining": 80}
    assert event["requests"][0]["allocated_block_ids"] == [[3]]
    assert event["requests"][0]["prefix_cached_tokens"] == 16

    failure = scheduler._make_scheduler_trace_allocation_failure(
        phase="running",
        request_id="A",
        num_new_tokens=32,
        num_computed_tokens=48,
        token_budget_remaining=64,
        input_budget_remaining=40,
    )
    assert failure == {
        "phase": "running",
        "request_id": "A",
        "num_new_tokens": 32,
        "num_computed_tokens": 48,
        "num_free_blocks": 7,
        "token_budget_remaining": 64,
        "input_budget_remaining": 40,
        "requires_kv_delivery": True,
        "preempted_request_id": None,
        "preempted_num_computed_tokens": None,
    }
