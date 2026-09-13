# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "vllm" / "v1" / "worker" / "gpu" / "model_runner.py"
SCHEDULER_PATH = ROOT / "vllm" / "v1" / "core" / "sched" / "scheduler.py"


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
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


def test_model_runner_trace_lifecycle_is_wired() -> None:
    init = _method(RUNNER_PATH, "GPUModelRunner", "__init__")
    prepare = _method(RUNNER_PATH, "GPUModelRunner", "prepare_inputs")
    shutdown = _method(RUNNER_PATH, "GPUModelRunner", "shutdown")

    assert "create_model_runner_trace_writer" in _called_names(init)
    assert "make_model_runner_batch_event" in _called_names(prepare)
    assert "record" in _called_names(prepare)
    assert "close" in _called_names(shutdown)


def test_scheduler_correlation_id_is_trace_guarded() -> None:
    schedule = _method(SCHEDULER_PATH, "Scheduler", "schedule")
    guarded_assignment = False
    for node in ast.walk(schedule):
        if not isinstance(node, ast.If):
            continue
        if "trace_writer is not None" not in ast.unparse(node.test):
            continue
        guarded_assignment = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "scheduler_trace_step_id"
                for target in child.targets
            )
            for child in node.body
        )
        if guarded_assignment:
            break

    assert guarded_assignment


def test_runner_reads_only_cpu_trace_surfaces() -> None:
    prepare = _method(RUNNER_PATH, "GPUModelRunner", "prepare_inputs")
    calls = _called_names(prepare)
    assert "make_model_runner_batch_event" in calls

    source = ast.unparse(prepare)
    event_call = source[source.index("make_model_runner_batch_event") :]
    event_call = event_call[: event_call.index("return pcp.maybe_partition_pcp_batch")]
    assert ".item(" not in event_call
    assert ".cpu(" not in event_call
    assert ".tolist(" not in event_call
    assert "synchronize(" not in event_call
