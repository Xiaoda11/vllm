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


def _named_call(method: ast.FunctionDef, name: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    )


def _assert_trace_call_is_cpu_only(call: ast.Call) -> None:
    source = ast.unparse(call)
    for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize("):
        assert forbidden not in source


def test_model_runner_trace_lifecycle_is_wired() -> None:
    init = _method(RUNNER_PATH, "GPUModelRunner", "__init__")
    prepare = _method(RUNNER_PATH, "GPUModelRunner", "prepare_inputs")
    sample = _method(RUNNER_PATH, "GPUModelRunner", "sample")
    shutdown = _method(RUNNER_PATH, "GPUModelRunner", "shutdown")

    assert "create_model_runner_trace_writer" in _called_names(init)
    assert "make_model_runner_batch_event" in _called_names(prepare)
    assert "make_sampler_batch_shard_event" in _called_names(sample)
    assert "record" in _called_names(prepare)
    assert "record" in _called_names(sample)
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


def test_runner_correlation_stays_on_batch_instances() -> None:
    prepare = _method(RUNNER_PATH, "GPUModelRunner", "prepare_inputs")
    sample = _method(RUNNER_PATH, "GPUModelRunner", "sample")
    prepare_source = ast.unparse(prepare)
    sample_source = ast.unparse(sample)

    # The step id follows each InputBatch through optional PCP partitioning.
    assert "setattr(input_batch, 'scheduler_trace_step_id'" in prepare_source
    assert "setattr(prepared_batch, 'scheduler_trace_step_id'" in prepare_source
    assert "getattr(global_input_batch, 'scheduler_trace_step_id', 0)" in sample_source

    # Do not store an in-flight step id on GPUModelRunner itself: overlapping
    # execute/sample batches could otherwise overwrite each other's correlation.
    assert "self.scheduler_trace_step_id" not in prepare_source
    assert "self.scheduler_trace_step_id" not in sample_source


def test_runner_trace_event_arguments_are_cpu_only() -> None:
    prepare = _method(RUNNER_PATH, "GPUModelRunner", "prepare_inputs")
    sample = _method(RUNNER_PATH, "GPUModelRunner", "sample")

    batch_call = _named_call(prepare, "make_model_runner_batch_event")
    shard_call = _named_call(sample, "make_sampler_batch_shard_event")
    _assert_trace_call_is_cpu_only(batch_call)
    _assert_trace_call_is_cpu_only(shard_call)

    shard_source = ast.unparse(shard_call)
    # GPU-only shard-plan surfaces must not be pulled back just for tracing.
    assert "gathered_src_indices" not in shard_source
    assert "sorted_logits_indices" not in shard_source
    assert "local_idx_mapping" not in shard_source
