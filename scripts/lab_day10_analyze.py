#!/usr/bin/env python3
# ruff: noqa: E501
"""Validate and summarize Day 10 decode/prefill workload artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = [
    "run_id",
    "b_prompt_tokens",
    "token_budget",
    "a_first_token_before_b_submit",
    "mixed_prefill_steps",
    "b_prefill_schedule_pattern",
    "max_mixed_batch_tokens",
    "a_ttft_ms",
    "a_tpot_ms",
    "a_e2e_s",
    "b_ttft_ms",
    "b_tpot_ms",
    "b_e2e_s",
    "total_output_tokens_per_s",
    "a_itl_before_b_submit_mean_ms",
    "a_itl_during_b_prefill_mean_ms",
    "a_itl_during_b_prefill_p95_ms",
    "a_itl_during_b_decode_mean_ms",
    "a_itl_after_b_finish_mean_ms",
    "preemptions",
    "allocation_failures",
    "mrv2_input_shape_mismatches",
]

SEGMENT_FIELDS = [
    "run_id",
    "b_prompt_tokens",
    "token_budget",
    "segment",
    "count",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "max_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rounded_ms(value: float | None) -> float | None:
    return round(value * 1000, 3) if value is not None else None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": _rounded_ms(sum(values) / len(values) if values else None),
        "p50_ms": _rounded_ms(percentile(values, 0.50)),
        "p95_ms": _rounded_ms(percentile(values, 0.95)),
        "max_ms": _rounded_ms(max(values) if values else None),
    }


def _request_by_id(rows: list[dict[str, str]], request_id: str) -> dict[str, str]:
    matches = [row for row in rows if row["request_id"] == request_id]
    if len(matches) != 1:
        raise ValueError(f"expected one request timing row for {request_id}")
    return matches[0]


def _trace_request(step: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    matches = [
        request
        for request in step.get("requests", [])
        if request["request_id"].startswith(prefix)
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple {prefix} requests in Scheduler step")
    return matches[0] if matches else None


def _format_pattern(values: list[int]) -> str:
    counts = Counter(values)
    return "|".join(f"{value}x{counts[value]}" for value in sorted(counts))


def analyze_run(
    run_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    request_rows = read_csv(run_directory / "request_timing.csv")
    token_rows = read_csv(run_directory / "token_timing.csv")
    scheduler_steps = [
        row
        for row in read_jsonl(run_directory / "scheduler_trace.jsonl")
        if row.get("event") == "scheduler_step"
    ]
    runner_rows = read_jsonl(run_directory / "scheduler_trace.mrv2.jsonl")
    runner_inputs = {
        int(row["step_id"]): row
        for row in runner_rows
        if row.get("event") == "model_runner_inputs" and int(row["step_id"]) > 0
    }

    if metadata["status"] != "passed":
        raise ValueError(f"{run_directory}: workload did not pass")
    effective = metadata["effective_scenario"]
    token_budget = int(effective["engine"]["max_num_batched_tokens"])
    request_specs = {item["request_id"]: item for item in effective["requests"]}
    b_prompt_tokens = int(request_specs["B"]["prompt_tokens"])
    a_timing = _request_by_id(request_rows, "A")
    b_timing = _request_by_id(request_rows, "B")
    for request_id, timing in (("A", a_timing), ("B", b_timing)):
        expected = int(request_specs[request_id]["output_tokens"])
        if (
            timing["status"] != "passed"
            or int(timing["actual_output_tokens"]) != expected
        ):
            raise ValueError(f"{run_directory}: {request_id} output validation failed")

    b_submitted_s = float(b_timing["submitted_s"])
    b_first_token_s = float(b_timing["first_token_s"])
    b_finished_s = float(b_timing["finished_s"])
    a_before_b = float(a_timing["first_token_s"]) < b_submitted_s
    if not a_before_b:
        raise ValueError(f"{run_directory}: A did not decode before B submission")

    mixed_prefill_steps: list[dict[str, Any]] = []
    preemptions = 0
    allocation_failures = 0
    shape_mismatches = 0
    for step in scheduler_steps:
        if step["token_budget"]["scheduled"] > token_budget:
            raise ValueError(f"{run_directory}: step exceeds token budget")
        preemptions += len(step.get("preempted_request_ids", []))
        allocation_failures += len(step.get("allocation_failures", []))
        a_request = _trace_request(step, "A-")
        b_request = _trace_request(step, "B-")
        b_before = b_request.get("before") if b_request else None
        if (
            a_request
            and b_request
            and b_before
            and a_request["num_scheduled_tokens"] == 1
            and b_before["num_computed_tokens"] < b_prompt_tokens
        ):
            mixed_prefill_steps.append(step)
        runner = runner_inputs.get(int(step["step_id"]))
        scheduled = int(step["token_budget"]["scheduled"])
        if scheduled == 0:
            continue
        if runner is None:
            shape_mismatches += 1
            continue
        input_tokens = int(runner["input_tensors"]["input_ids"]["shape"][0])
        batch_tokens = int(runner["batch"]["num_tokens"])
        query_end = int(runner["cpu_inputs"]["query_start_loc"][-1])
        if (
            scheduled != input_tokens
            or scheduled != batch_tokens
            or scheduled != query_end
        ):
            shape_mismatches += 1

    if not mixed_prefill_steps:
        raise ValueError(f"{run_directory}: no decode/prefill mixed step")
    if shape_mismatches:
        raise ValueError(f"{run_directory}: {shape_mismatches} MRV2 shape mismatches")

    b_prefill_scheduled = [
        int(_trace_request(step, "B-")["num_scheduled_tokens"])
        for step in mixed_prefill_steps
    ]
    mixed_batch_tokens = [
        int(step["token_budget"]["scheduled"]) for step in mixed_prefill_steps
    ]

    segment_values: dict[str, list[float]] = {
        "before_b_submit": [],
        "during_b_prefill": [],
        "during_b_decode": [],
        "after_b_finish": [],
    }
    for row in token_rows:
        if row["request_id"] != "A" or not row["single_token_itl_s"]:
            continue
        emitted_s = float(row["emitted_s"])
        itl_s = float(row["single_token_itl_s"])
        if emitted_s < b_submitted_s:
            segment = "before_b_submit"
        elif emitted_s < b_first_token_s:
            segment = "during_b_prefill"
        elif emitted_s < b_finished_s:
            segment = "during_b_decode"
        else:
            segment = "after_b_finish"
        segment_values[segment].append(itl_s)

    segment_rows = []
    for segment, values in segment_values.items():
        segment_rows.append(
            {
                "run_id": metadata["run_id"],
                "b_prompt_tokens": b_prompt_tokens,
                "token_budget": token_budget,
                "segment": segment,
                **summarize_values(values),
            }
        )
    segment_stats = {row["segment"]: row for row in segment_rows}
    total_output_tokens = sum(int(row["actual_output_tokens"]) for row in request_rows)
    makespan_s = max(float(row["finished_s"]) for row in request_rows)
    summary = {
        "run_id": metadata["run_id"],
        "b_prompt_tokens": b_prompt_tokens,
        "token_budget": token_budget,
        "a_first_token_before_b_submit": a_before_b,
        "mixed_prefill_steps": len(mixed_prefill_steps),
        "b_prefill_schedule_pattern": _format_pattern(b_prefill_scheduled),
        "max_mixed_batch_tokens": max(mixed_batch_tokens),
        "a_ttft_ms": _rounded_ms(float(a_timing["ttft_s"])),
        "a_tpot_ms": _rounded_ms(float(a_timing["tpot_s"])),
        "a_e2e_s": float(a_timing["e2e_s"]),
        "b_ttft_ms": _rounded_ms(float(b_timing["ttft_s"])),
        "b_tpot_ms": _rounded_ms(float(b_timing["tpot_s"])),
        "b_e2e_s": float(b_timing["e2e_s"]),
        "total_output_tokens_per_s": round(total_output_tokens / makespan_s, 3),
        "a_itl_before_b_submit_mean_ms": segment_stats["before_b_submit"]["mean_ms"],
        "a_itl_during_b_prefill_mean_ms": segment_stats["during_b_prefill"]["mean_ms"],
        "a_itl_during_b_prefill_p95_ms": segment_stats["during_b_prefill"]["p95_ms"],
        "a_itl_during_b_decode_mean_ms": segment_stats["during_b_decode"]["mean_ms"],
        "a_itl_after_b_finish_mean_ms": segment_stats["after_b_finish"]["mean_ms"],
        "preemptions": preemptions,
        "allocation_failures": allocation_failures,
        "mrv2_input_shape_mismatches": shape_mismatches,
    }
    return summary, segment_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tradeoff_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 960, 360
    colors = {8192: "#2563eb", 16384: "#dc2626"}
    metrics = [
        ("b_ttft_ms", "B TTFT (ms)"),
        ("a_itl_during_b_prefill_mean_ms", "A ITL during B prefill (ms)"),
        ("total_output_tokens_per_s", "Output throughput (token/s)"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#111827}.title{font-size:13px;font-weight:bold}.axis{stroke:#9ca3af}.grid{stroke:#e5e7eb}</style>",
    ]
    panel_width = 300
    for panel, (field, title) in enumerate(metrics):
        x0 = 45 + panel * 315
        y0, plot_w, plot_h = 45, 235, 235
        values = [float(row[field]) for row in rows]
        lower, upper = min(values), max(values)
        padding = (upper - lower) * 0.12 or max(abs(upper) * 0.1, 1.0)
        lower, upper = lower - padding, upper + padding
        parts.append(f'<text class="title" x="{x0}" y="24">{title}</text>')
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>'
        )
        for tick in range(3):
            value = lower + (upper - lower) * tick / 2
            y = y0 + plot_h - plot_h * tick / 2
            parts.append(
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>'
            )
            parts.append(
                f'<text x="{x0 - 5}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>'
            )
        for index, budget in enumerate((2048, 4096, 8192)):
            x = x0 + index * plot_w / 2
            parts.append(
                f'<text x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle">{budget}</text>'
            )
        for prompt_tokens in (8192, 16384):
            series = sorted(
                (row for row in rows if int(row["b_prompt_tokens"]) == prompt_tokens),
                key=lambda row: int(row["token_budget"]),
            )
            points = []
            for index, row in enumerate(series):
                x = x0 + index * plot_w / 2
                value = float(row[field])
                y = y0 + plot_h - (value - lower) / (upper - lower) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[prompt_tokens]}"/>'
                )
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[prompt_tokens]}" stroke-width="2"/>'
            )
        parts.append(
            f'<text x="{x0 + plot_w / 2}" y="{height - 24}" text-anchor="middle">global token budget</text>'
        )
        if panel == 2:
            parts.append(
                f'<text x="{x0 + panel_width - 50}" y="48" fill="{colors[8192]}">B=8K</text>'
            )
            parts.append(
                f'<text x="{x0 + panel_width - 50}" y="64" fill="{colors[16384]}">B=16K</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = []
    segments = []
    for run_directory in args.run_directories:
        summary, segment_rows = analyze_run(run_directory)
        summaries.append(summary)
        segments.extend(segment_rows)
    summaries.sort(key=lambda row: (row["b_prompt_tokens"], row["token_budget"]))
    segments.sort(
        key=lambda row: (row["b_prompt_tokens"], row["token_budget"], row["segment"])
    )
    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_csv(
        args.output_directory / "day10_matrix_summary.csv", summaries, SUMMARY_FIELDS
    )
    write_csv(
        args.output_directory / "day10_itl_segments.csv", segments, SEGMENT_FIELDS
    )
    write_tradeoff_svg(args.output_directory / "day10_tradeoffs.svg", summaries)
    print(json.dumps({"runs": len(summaries), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
