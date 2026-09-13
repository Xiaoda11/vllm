# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_event_module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "vllm"
        / "v1"
        / "core"
        / "sched"
        / "model_runner_trace_event.py"
    )
    spec = importlib.util.spec_from_file_location("model_runner_trace_event", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


event_mod = _load_event_module()
make_model_runner_batch_event = event_mod.make_model_runner_batch_event


def test_model_runner_event_preserves_v026_join_fields() -> None:
    event = make_model_runner_batch_event(
        step_id=7,
        request_ids=["A", "B"],
        persistent_rows=[2, 9],
        num_scheduled_tokens=[1, 4095],
        scheduler_total_num_scheduled_tokens=4096,
        runner_num_tokens=4096,
        model_num_tokens_after_padding=4096,
        num_reqs_after_padding=2,
        has_prefill=True,
        cudagraph_mode="NONE",
        adaptive_verification_active=False,
        batch_sharded_sampling_enabled=False,
        timestamp_ns=123,
    )

    assert event["schema_version"] == 1
    assert event["event"] == "model_runner_batch"
    assert event["step_id"] == 7
    assert event["timestamp_ns"] == 123
    assert event["request_ids"] == ["A", "B"]
    assert event["persistent_rows"] == [2, 9]
    assert event["num_scheduled_tokens"] == [1, 4095]
    assert event["request_id_to_persistent_row"] == {"A": 2, "B": 9}
    assert event["token_counts"] == {
        "scheduler_logical": 4096,
        "runner_effective_unpadded": 4096,
        "model_input_after_padding": 4096,
        "scheduler_to_runner_delta": 0,
        "graph_padding": 0,
    }


def test_model_runner_event_distinguishes_trim_and_graph_padding() -> None:
    event = make_model_runner_batch_event(
        step_id=11,
        request_ids=["decode"],
        persistent_rows=[4],
        num_scheduled_tokens=[5],
        scheduler_total_num_scheduled_tokens=5,
        runner_num_tokens=3,
        model_num_tokens_after_padding=8,
        num_reqs_after_padding=4,
        has_prefill=False,
        cudagraph_mode="FULL",
        adaptive_verification_active=True,
        batch_sharded_sampling_enabled=True,
        timestamp_ns=456,
    )

    assert event["token_counts"]["scheduler_to_runner_delta"] == 2
    assert event["token_counts"]["graph_padding"] == 5
    assert event["request_counts"] == {
        "logical": 1,
        "model_input_after_padding": 4,
    }
    assert event["adaptive_verification_active"] is True
    assert event["batch_sharded_sampling_enabled"] is True
    assert event["cudagraph_mode"] == "FULL"


def test_model_runner_event_rejects_inconsistent_cpu_shapes() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        make_model_runner_batch_event(
            step_id=1,
            request_ids=["A", "B"],
            persistent_rows=[0],
            num_scheduled_tokens=[1, 1],
            scheduler_total_num_scheduled_tokens=2,
            runner_num_tokens=2,
            model_num_tokens_after_padding=2,
            num_reqs_after_padding=2,
            has_prefill=False,
            cudagraph_mode="NONE",
            adaptive_verification_active=False,
            batch_sharded_sampling_enabled=False,
        )
