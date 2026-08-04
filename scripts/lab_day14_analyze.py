#!/usr/bin/env python3
"""Validate and aggregate Day 14 waiting HOL Baseline/Modified runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--modified-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def relative_change_percent(baseline: float, modified: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (modified / baseline - 1.0) * 100.0


def jain_index(values: list[float]) -> float:
    if not values or any(value < 0 for value in values):
        raise ValueError("Jain values must be non-empty and non-negative")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 1.0
    return sum(values) ** 2 / denominator


def median(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def _request_by_prefix(step: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    matches = [
        request
        for request in step.get("requests", [])
        if request["request_id"].startswith(f"{prefix}-")
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple requests match {prefix}")
    return matches[0] if matches else None


def _queue_index(queue: list[str], prefix: str) -> int | None:
    matches = [i for i, request_id in enumerate(queue) if request_id.startswith(prefix)]
    if len(matches) > 1:
        raise ValueError(f"multiple queue entries match {prefix}")
    return matches[0] if matches else None


def analyze_run(run_directory: Path, expected_mode: str) -> dict[str, Any]:
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    if metadata["status"] != "passed":
        raise ValueError(f"{run_directory}: workload did not pass")

    scenario = metadata["effective_scenario"]
    if scenario["scenario_id"] != "D13" or len(scenario["requests"]) != 3:
        raise ValueError(f"{run_directory}: expected the three-request D13 scenario")
    engine = scenario["engine"]
    enabled = bool(engine.get("scheduler_allow_waiting_bypass", False))
    actual_mode = "modified" if enabled else "baseline"
    if actual_mode != expected_mode:
        raise ValueError(
            f"{run_directory}: expected {expected_mode}, observed {actual_mode}"
        )
    if not engine.get("scheduler_reserve_full_isl", True):
        raise ValueError(f"{run_directory}: full-ISL reservation must remain enabled")

    specs = {request["request_id"]: request for request in scenario["requests"]}
    if set(specs) != {"A", "B", "C"}:
        raise ValueError(f"{run_directory}: expected request ids A/B/C")
    block_size = 16
    required_c_blocks = math.ceil(int(specs["C"]["prompt_tokens"]) / block_size)

    results = {result["request_id"]: result for result in metadata["request_results"]}
    if set(results) != {"A", "B", "C"}:
        raise ValueError(f"{run_directory}: missing request results")
    if any(result["status"] != "passed" for result in results.values()):
        raise ValueError(f"{run_directory}: request validation failed")
    if not float(results["B"]["submitted_s"]) < float(results["C"]["submitted_s"]):
        raise ValueError(f"{run_directory}: B was not submitted before C")

    scheduler_steps = [
        row
        for row in read_jsonl(run_directory / "scheduler_trace.jsonl")
        if row.get("event") == "scheduler_step"
    ]
    runner_inputs = {
        int(row["step_id"]): row
        for row in read_jsonl(run_directory / "scheduler_trace.mrv2.jsonl")
        if row.get("event") == "model_runner_inputs" and int(row["step_id"]) > 0
    }

    first_scheduled: dict[str, int | None] = {key: None for key in ("A", "B", "C")}
    waiting_failures = 0
    preemptions = 0
    shape_mismatches = 0
    hol_witness = False
    for step in scheduler_steps:
        step_id = int(step["step_id"])
        scheduled = int(step["token_budget"]["scheduled"])
        preemptions += len(step.get("preempted_request_ids", []))
        for prefix in first_scheduled:
            request = _request_by_prefix(step, prefix)
            if (
                request is not None
                and int(request["num_scheduled_tokens"]) > 0
                and first_scheduled[prefix] is None
            ):
                first_scheduled[prefix] = step_id

        for failure in step.get("allocation_failures", []):
            if failure["phase"] != "waiting" or not failure["request_id"].startswith(
                "B-"
            ):
                continue
            waiting_failures += 1
            queues = step["queues"]
            b_index = _queue_index(queues["waiting_before"], "B-")
            c_index = _queue_index(queues["waiting_before"], "C-")
            if b_index is None:
                b_index = _queue_index(queues["skipped_waiting_before"], "B-")
            free_blocks = int(failure["num_free_blocks"])
            if (
                b_index is not None
                and c_index is not None
                and free_blocks >= required_c_blocks
            ):
                hol_witness = True

        if scheduled == 0:
            continue
        runner = runner_inputs.get(step_id)
        if runner is None:
            shape_mismatches += 1
            continue
        observed = (
            int(runner["input_tensors"]["input_ids"]["shape"][0]),
            int(runner["batch"]["num_tokens"]),
            int(runner["cpu_inputs"]["query_start_loc"][-1]),
        )
        shape_mismatches += sum(value != scheduled for value in observed)

    if any(value is None for value in first_scheduled.values()):
        raise ValueError(f"{run_directory}: a request was never scheduled")
    if not hol_witness:
        raise ValueError(f"{run_directory}: no valid B-blocks-C-fits HOL witness")
    if shape_mismatches:
        raise ValueError(f"{run_directory}: MRV2 shape mismatch")

    submitted = [float(result["submitted_s"]) for result in results.values()]
    finished = [float(result["finished_s"]) for result in results.values()]
    makespan_s = max(finished) - min(submitted)
    total_output_tokens = sum(
        int(result["actual_output_tokens"]) for result in results.values()
    )
    b_ttft_s = float(results["B"]["ttft_s"])
    c_ttft_s = float(results["C"]["ttft_s"])
    return {
        "run_id": metadata["run_id"],
        "mode": actual_mode,
        "a_ttft_ms": round(float(results["A"]["ttft_s"]) * 1000, 3),
        "b_ttft_ms": round(b_ttft_s * 1000, 3),
        "c_ttft_ms": round(c_ttft_s * 1000, 3),
        "a_e2e_ms": round(float(results["A"]["e2e_s"]) * 1000, 3),
        "b_e2e_ms": round(float(results["B"]["e2e_s"]) * 1000, 3),
        "c_e2e_ms": round(float(results["C"]["e2e_s"]) * 1000, 3),
        "makespan_ms": round(makespan_s * 1000, 3),
        "output_tokens_per_s": round(total_output_tokens / makespan_s, 3),
        "b_c_ttft_jain": round(jain_index([b_ttft_s, c_ttft_s]), 6),
        "a_first_scheduled_step": first_scheduled["A"],
        "b_first_scheduled_step": first_scheduled["B"],
        "c_first_scheduled_step": first_scheduled["C"],
        "waiting_allocation_failures": waiting_failures,
        "preemptions": preemptions,
        "mrv2_input_shape_mismatches": shape_mismatches,
        "hol_witness": hol_witness,
    }


def summarize(
    baseline: list[dict[str, Any]], modified: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = [
        "a_ttft_ms",
        "b_ttft_ms",
        "c_ttft_ms",
        "a_e2e_ms",
        "b_e2e_ms",
        "c_e2e_ms",
        "makespan_ms",
        "output_tokens_per_s",
        "b_c_ttft_jain",
        "b_first_scheduled_step",
        "c_first_scheduled_step",
        "waiting_allocation_failures",
    ]
    baseline_medians = {field: median(baseline, field) for field in fields}
    modified_medians = {field: median(modified, field) for field in fields}
    changes = {
        field: round(
            relative_change_percent(baseline_medians[field], modified_medians[field]),
            3,
        )
        for field in fields
        if baseline_medians[field] != 0
    }
    return {
        "run_counts": {"baseline": len(baseline), "modified": len(modified)},
        "baseline_medians": baseline_medians,
        "modified_medians": modified_medians,
        "modified_vs_baseline_percent": changes,
        "repeat_gate": {
            "at_least_three_runs_per_mode": len(baseline) >= 3 and len(modified) >= 3,
            "all_runs_have_hol_witness": all(
                bool(row["hol_witness"]) for row in baseline + modified
            ),
            "all_runs_complete_without_preemption_or_shape_mismatch": all(
                int(row["preemptions"]) == 0
                and int(row["mrv2_input_shape_mismatches"]) == 0
                for row in baseline + modified
            ),
            "modified_schedules_c_before_b": all(
                int(row["c_first_scheduled_step"])
                < int(row["b_first_scheduled_step"])
                for row in modified
            ),
            "baseline_schedules_b_before_c": all(
                int(row["b_first_scheduled_step"])
                < int(row["c_first_scheduled_step"])
                for row in baseline
            ),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline = [analyze_run(path, "baseline") for path in args.baseline_runs]
    modified = [analyze_run(path, "modified") for path in args.modified_runs]
    summary = summarize(baseline, modified)

    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_directory / "day14_hol_runs.csv", baseline + modified)
    (args.output_directory / "day14_hol_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
