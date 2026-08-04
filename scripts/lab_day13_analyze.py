#!/usr/bin/env python3
"""Validate and aggregate Day 13 Prefill-quantum Gate A/B runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.lab_day10_analyze import analyze_run as analyze_mixed_run
else:
    from lab_day10_analyze import analyze_run as analyze_mixed_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--pure-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def relative_change_percent(baseline: float, tuned: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (tuned / baseline - 1.0) * 100.0


def _request_by_prefix(step: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    matches = [
        request
        for request in step.get("requests", [])
        if request["request_id"].startswith(prefix)
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple requests match {prefix}")
    return matches[0] if matches else None


def _format_pattern(values: list[int]) -> str:
    counts = Counter(values)
    return "|".join(f"{value}x{counts[value]}" for value in sorted(counts))


def analyze_pure_run(run_directory: Path) -> dict[str, Any]:
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    if metadata["status"] != "passed":
        raise ValueError(f"{run_directory}: workload did not pass")

    effective = metadata["effective_scenario"]
    if effective["scenario_id"] != "S2" or len(effective["requests"]) != 1:
        raise ValueError(f"{run_directory}: expected one S2 request")
    request_spec = effective["requests"][0]
    request_id = request_spec["request_id"]
    prompt_tokens = int(request_spec["prompt_tokens"])
    output_tokens = int(request_spec["output_tokens"])
    threshold = int(effective["engine"]["long_prefill_token_threshold"])

    results = metadata["request_results"]
    if len(results) != 1:
        raise ValueError(f"{run_directory}: expected one request result")
    result = results[0]
    if (
        result["status"] != "passed"
        or int(result["actual_prompt_tokens"]) != prompt_tokens
        or int(result["actual_output_tokens"]) != output_tokens
    ):
        raise ValueError(f"{run_directory}: request validation failed")

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

    prefill_chunks: list[int] = []
    preemptions = 0
    allocation_failures = 0
    shape_mismatches = 0
    for step in scheduler_steps:
        scheduled = int(step["token_budget"]["scheduled"])
        preemptions += len(step.get("preempted_request_ids", []))
        allocation_failures += len(step.get("allocation_failures", []))
        request = _request_by_prefix(step, f"{request_id}-")
        if request is not None:
            before = request.get("before")
            if before and int(before["num_computed_tokens"]) < prompt_tokens:
                prefill_chunks.append(int(request["num_scheduled_tokens"]))
        if scheduled == 0:
            continue
        runner = runner_inputs.get(int(step["step_id"]))
        if runner is None:
            shape_mismatches += 1
            continue
        if scheduled != int(runner["input_tensors"]["input_ids"]["shape"][0]):
            shape_mismatches += 1
        if scheduled != int(runner["batch"]["num_tokens"]):
            shape_mismatches += 1
        if scheduled != int(runner["cpu_inputs"]["query_start_loc"][-1]):
            shape_mismatches += 1

    if not prefill_chunks:
        raise ValueError(f"{run_directory}: no prefill chunks")
    if shape_mismatches:
        raise ValueError(f"{run_directory}: MRV2 shape mismatch")

    e2e_s = float(result["e2e_s"])
    return {
        "run_id": metadata["run_id"],
        "workload": "pure_16k",
        "threshold": threshold,
        "ttft_ms": round(float(result["ttft_s"]) * 1000, 3),
        "e2e_ms": round(e2e_s * 1000, 3),
        "input_tokens_per_s": round(prompt_tokens / e2e_s, 3),
        "prefill_steps": len(prefill_chunks),
        "prefill_pattern": _format_pattern(prefill_chunks),
        "preemptions": preemptions,
        "allocation_failures": allocation_failures,
        "mrv2_input_shape_mismatches": shape_mismatches,
    }


def median_for(rows: list[dict[str, Any]], threshold: int, field: str) -> float:
    values = [float(row[field]) for row in rows if int(row["threshold"]) == threshold]
    if not values:
        raise ValueError(f"no rows for threshold={threshold}, field={field}")
    return statistics.median(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    mixed_rows = []
    for run_directory in args.mixed_runs:
        row, _ = analyze_mixed_run(run_directory)
        metadata = json.loads((run_directory / "run_metadata.json").read_text())
        mixed_rows.append(
            {
                "run_id": row["run_id"],
                "workload": "mixed_16k",
                "threshold": int(
                    metadata["effective_scenario"]["engine"][
                        "long_prefill_token_threshold"
                    ]
                ),
                "a_prefill_itl_mean_ms": row["a_itl_during_b_prefill_mean_ms"],
                "a_prefill_itl_p95_ms": row["a_itl_during_b_prefill_p95_ms"],
                "b_ttft_ms": row["b_ttft_ms"],
                "b_e2e_ms": round(float(row["b_e2e_s"]) * 1000, 3),
                "output_tokens_per_s": row["total_output_tokens_per_s"],
                "mixed_prefill_steps": row["mixed_prefill_steps"],
                "prefill_pattern": row["b_prefill_schedule_pattern"],
                "preemptions": row["preemptions"],
                "allocation_failures": row["allocation_failures"],
                "mrv2_input_shape_mismatches": row["mrv2_input_shape_mismatches"],
            }
        )
    pure_rows = [analyze_pure_run(path) for path in args.pure_runs]

    mixed_p95_0 = median_for(mixed_rows, 0, "a_prefill_itl_p95_ms")
    mixed_p95_2048 = median_for(mixed_rows, 2048, "a_prefill_itl_p95_ms")
    pure_ttft_0 = median_for(pure_rows, 0, "ttft_ms")
    pure_ttft_2048 = median_for(pure_rows, 2048, "ttft_ms")
    pure_e2e_0 = median_for(pure_rows, 0, "e2e_ms")
    pure_e2e_2048 = median_for(pure_rows, 2048, "e2e_ms")
    pure_input_0 = median_for(pure_rows, 0, "input_tokens_per_s")
    pure_input_2048 = median_for(pure_rows, 2048, "input_tokens_per_s")
    summary = {
        "mixed_16k": {
            "itl_p95_baseline_median_ms": mixed_p95_0,
            "itl_p95_tuned_median_ms": mixed_p95_2048,
            "itl_p95_change_percent": round(
                relative_change_percent(mixed_p95_0, mixed_p95_2048), 3
            ),
        },
        "pure_16k": {
            "ttft_change_percent": round(
                relative_change_percent(pure_ttft_0, pure_ttft_2048), 3
            ),
            "e2e_change_percent": round(
                relative_change_percent(pure_e2e_0, pure_e2e_2048), 3
            ),
            "input_throughput_change_percent": round(
                relative_change_percent(pure_input_0, pure_input_2048), 3
            ),
        },
    }
    summary["gate"] = {
        "mixed_benefit_at_least_30_percent": summary["mixed_16k"][
            "itl_p95_change_percent"
        ]
        <= -30.0,
        "non_mixed_regression_over_5_percent": (
            summary["pure_16k"]["ttft_change_percent"] > 5.0
            or summary["pure_16k"]["e2e_change_percent"] > 5.0
            or summary["pure_16k"]["input_throughput_change_percent"] < -5.0
        ),
    }
    summary["gate"]["code_gate_triggered"] = all(summary["gate"].values())

    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_directory / "day13_mixed_runs.csv", mixed_rows)
    write_csv(args.output_directory / "day13_pure_runs.csv", pure_rows)
    (args.output_directory / "day13_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
