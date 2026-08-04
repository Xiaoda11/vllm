#!/usr/bin/env python3
"""Validate the Day 14 waiting-bypass burst fairness pair."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

if __package__:
    from scripts.lab_day14_analyze import (
        jain_index,
        read_jsonl,
        relative_change_percent,
    )
else:
    from lab_day14_analyze import jain_index, read_jsonl, relative_change_percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--modified-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scenario-id", default="D14B", choices=("D14B", "D15"))
    return parser.parse_args()


def _first_scheduled_step(steps: list[dict[str, Any]], prefix: str) -> int:
    for step in steps:
        for request in step.get("requests", []):
            if request["request_id"].startswith(f"{prefix}-") and int(
                request["num_scheduled_tokens"]
            ) > 0:
                return int(step["step_id"])
    raise ValueError(f"request {prefix} was never scheduled")


def analyze_burst_run(
    run_directory: Path, expected_mode: str, scenario_id: str = "D14B"
) -> dict[str, Any]:
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    if metadata["status"] != "passed":
        raise ValueError(f"{run_directory}: workload did not pass")
    scenario = metadata["effective_scenario"]
    if scenario["scenario_id"] != scenario_id:
        raise ValueError(f"{run_directory}: expected {scenario_id}")
    enabled = bool(
        scenario["engine"].get("scheduler_allow_waiting_bypass", False)
    )
    mode = "modified" if enabled else "baseline"
    if mode != expected_mode:
        raise ValueError(f"{run_directory}: expected {expected_mode}, observed {mode}")
    if not scenario["engine"].get("scheduler_reserve_full_isl", True):
        raise ValueError(f"{run_directory}: full-ISL reservation must stay enabled")

    results = {result["request_id"]: result for result in metadata["request_results"]}
    expected_ids = {"A", "B", *(f"C{i}" for i in range(1, 9))}
    if set(results) != expected_ids:
        raise ValueError(f"{run_directory}: missing burst request results")
    if any(result["status"] != "passed" for result in results.values()):
        raise ValueError(f"{run_directory}: a request failed")
    if any(
        float(results["B"]["submitted_s"]) >= float(results[f"C{i}"]["submitted_s"])
        for i in range(1, 9)
    ):
        raise ValueError(f"{run_directory}: B must be submitted before every C request")

    steps = [
        row
        for row in read_jsonl(run_directory / "scheduler_trace.jsonl")
        if row.get("event") == "scheduler_step"
    ]
    runner_inputs = {
        int(row["step_id"]): row
        for row in read_jsonl(run_directory / "scheduler_trace.mrv2.jsonl")
        if row.get("event") == "model_runner_inputs" and int(row["step_id"]) > 0
    }
    first_steps = {
        request_id: _first_scheduled_step(steps, request_id)
        for request_id in expected_ids
    }
    failures = 0
    preemptions = 0
    shape_mismatches = 0
    hol_witness = False
    for step in steps:
        scheduled = int(step["token_budget"]["scheduled"])
        preemptions += len(step.get("preempted_request_ids", []))
        for failure in step.get("allocation_failures", []):
            failures += 1
            if failure["phase"] != "waiting" or not failure["request_id"].startswith(
                "B-"
            ):
                continue
            queues = step["queues"]
            all_waiting = (
                queues["skipped_waiting_before"] + queues["waiting_before"]
            )
            if any(request_id.startswith("C") for request_id in all_waiting) and int(
                failure["num_free_blocks"]
            ) >= 64:
                hol_witness = True
        if scheduled == 0:
            continue
        runner = runner_inputs.get(int(step["step_id"]))
        if runner is None:
            shape_mismatches += 1
            continue
        observed = (
            int(runner["input_tensors"]["input_ids"]["shape"][0]),
            int(runner["batch"]["num_tokens"]),
            int(runner["cpu_inputs"]["query_start_loc"][-1]),
        )
        shape_mismatches += sum(value != scheduled for value in observed)
    if not hol_witness:
        raise ValueError(f"{run_directory}: missing HOL witness")
    if shape_mismatches:
        raise ValueError(f"{run_directory}: MRV2 shape mismatch")

    c_ttft_s = [float(results[f"C{i}"]["ttft_s"]) for i in range(1, 9)]
    submitted = [float(result["submitted_s"]) for result in results.values()]
    finished = [float(result["finished_s"]) for result in results.values()]
    makespan_s = max(finished) - min(submitted)
    total_output_tokens = sum(
        int(result["actual_output_tokens"]) for result in results.values()
    )
    return {
        "run_id": metadata["run_id"],
        "mode": mode,
        "a_ttft_ms": round(float(results["A"]["ttft_s"]) * 1000, 3),
        "b_ttft_ms": round(float(results["B"]["ttft_s"]) * 1000, 3),
        "b_e2e_ms": round(float(results["B"]["e2e_s"]) * 1000, 3),
        "c1_ttft_ms": round(float(results["C1"]["ttft_s"]) * 1000, 3),
        "c_ttft_median_ms": round(statistics.median(c_ttft_s) * 1000, 3),
        "c_ttft_max_ms": round(max(c_ttft_s) * 1000, 3),
        "all_ttft_jain": round(
            jain_index([float(result["ttft_s"]) for result in results.values()]), 6
        ),
        "makespan_ms": round(makespan_s * 1000, 3),
        "output_tokens_per_s": round(total_output_tokens / makespan_s, 3),
        "b_first_scheduled_step": first_steps["B"],
        "c_first_scheduled_step_min": min(
            first_steps[f"C{i}"] for i in range(1, 9)
        ),
        "c_first_scheduled_step_max": max(
            first_steps[f"C{i}"] for i in range(1, 9)
        ),
        "allocation_failures": failures,
        "preemptions": preemptions,
        "mrv2_input_shape_mismatches": shape_mismatches,
        "hol_witness": hol_witness,
    }


def summarize_pair(
    baseline: dict[str, Any], modified: dict[str, Any]
) -> dict[str, Any]:
    fields = [
        "a_ttft_ms",
        "b_ttft_ms",
        "b_e2e_ms",
        "c1_ttft_ms",
        "c_ttft_median_ms",
        "c_ttft_max_ms",
        "all_ttft_jain",
        "makespan_ms",
        "output_tokens_per_s",
        "b_first_scheduled_step",
        "c_first_scheduled_step_min",
        "c_first_scheduled_step_max",
        "allocation_failures",
    ]
    changes = {
        field: round(
            relative_change_percent(float(baseline[field]), float(modified[field])),
            3,
        )
        for field in fields
        if float(baseline[field]) != 0
    }
    return {
        "baseline": baseline,
        "modified": modified,
        "modified_vs_baseline_percent": changes,
        "fairness_gate": {
            "both_runs_have_hol_witness": bool(
                baseline["hol_witness"] and modified["hol_witness"]
            ),
            "both_runs_complete_without_preemption_or_shape_mismatch": all(
                int(row["preemptions"]) == 0
                and int(row["mrv2_input_shape_mismatches"]) == 0
                for row in (baseline, modified)
            ),
            "modified_delays_b_first_schedule": int(
                modified["b_first_scheduled_step"]
            )
            > int(baseline["b_first_scheduled_step"]),
        },
    }


def main() -> None:
    args = parse_args()
    baseline = analyze_burst_run(
        args.baseline_run, "baseline", scenario_id=args.scenario_id
    )
    modified = analyze_burst_run(
        args.modified_run, "modified", scenario_id=args.scenario_id
    )
    summary = summarize_pair(baseline, modified)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    output_stem = args.scenario_id.lower() + "_burst"
    with (args.output_directory / f"{output_stem}_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline))
        writer.writeheader()
        writer.writerows([baseline, modified])
    (args.output_directory / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
