#!/usr/bin/env python3
"""Validate and compare bounded Day 16 PyTorch profiler windows."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.lab_day14_analyze import analyze_run, read_jsonl
except ModuleNotFoundError:
    from lab_day14_analyze import analyze_run, read_jsonl


EXECUTION_RE = re.compile(
    r"^execute_(?P<total_tokens>\d+)_context_(?P<context_requests>\d+)"
    r"\(sq(?P<context_tokens>\d+)sk(?P<context_kv_tokens>\d+)"
    r"sqsq\d+sqsk\d+\)_generation_(?P<generation_requests>\d+)"
    r"\(sq(?P<generation_tokens>\d+)sk(?P<generation_kv_tokens>\d+)"
    r"sqsq\d+sqsk\d+\)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--bounded-run", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def parse_execution_annotation(name: str) -> dict[str, int]:
    match = EXECUTION_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unrecognized execution annotation: {name}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    if (
        values["context_tokens"] + values["generation_tokens"]
        != values["total_tokens"]
    ):
        raise ValueError(f"execution token total is inconsistent: {name}")
    return values


def classify_execution(values: dict[str, int]) -> str:
    context = values["context_tokens"]
    generation = values["generation_tokens"]
    if context > 0 and generation > 0:
        return "prefill_decode_mixed"
    if context > 0:
        return "prefill_only"
    if values["generation_requests"] > 1:
        return "multi_decode"
    return "single_decode"


def _trace_path(run_directory: Path) -> Path:
    matches = sorted((run_directory / "torch_profile").glob("*.pt.trace.json.gz"))
    if len(matches) != 1:
        raise ValueError(f"{run_directory}: expected exactly one torch trace")
    return matches[0]


def _kernel_family(name: str) -> str:
    lowered = name.lower()
    if name == "kernel_unified_attention":
        return "attention"
    if "gemv" in lowered:
        return "gemv"
    if "gemm" in lowered or "cutlass::kernel" in lowered:
        return "gemm"
    return "other"


def analyze_profile(run_directory: Path, expected_mode: str) -> dict[str, Any]:
    scheduler = analyze_run(run_directory, expected_mode)
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    profiler_config = metadata["effective_scenario"]["engine"].get(
        "profiler_config", {}
    )
    if profiler_config.get("profiler") != "torch":
        raise ValueError(f"{run_directory}: torch profiler was not enabled")
    delay = int(profiler_config["delay_iterations"])
    max_iterations = int(profiler_config["max_iterations"])
    if delay < 0 or max_iterations <= 0:
        raise ValueError(f"{run_directory}: invalid bounded profile window")

    with gzip.open(_trace_path(run_directory), "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]

    cpu_annotations = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("execute_")
    ]
    if len(cpu_annotations) != max_iterations:
        raise ValueError(
            f"{run_directory}: expected {max_iterations} execution annotations, "
            f"observed {len(cpu_annotations)}"
        )
    gpu_annotations = {
        int(event["args"]["External id"]): event
        for event in events
        if event.get("cat") == "gpu_user_annotation"
        and str(event.get("name", "")).startswith("execute_")
    }

    scheduler_steps = {
        int(row["step_id"]): row
        for row in read_jsonl(run_directory / "scheduler_trace.jsonl")
        if row.get("event") == "scheduler_step"
    }
    rows: list[dict[str, Any]] = []
    for offset, annotation in enumerate(cpu_annotations):
        step_id = delay + offset
        step = scheduler_steps.get(step_id)
        if step is None:
            raise ValueError(f"{run_directory}: missing Scheduler step {step_id}")
        values = parse_execution_annotation(annotation["name"])
        scheduled_tokens = int(step["token_budget"]["scheduled"])
        if scheduled_tokens != values["total_tokens"]:
            raise ValueError(
                f"{run_directory}: step {step_id} scheduled {scheduled_tokens}, "
                f"profile annotation recorded {values['total_tokens']}"
            )
        external_id = int(annotation["args"]["External id"])
        gpu_annotation = gpu_annotations.get(external_id)
        if gpu_annotation is None:
            raise ValueError(
                f"{run_directory}: missing GPU annotation for step {step_id}"
            )
        rows.append(
            {
                "mode": expected_mode,
                "step_id": step_id,
                "classification": classify_execution(values),
                "scheduled_tokens": scheduled_tokens,
                "context_tokens": values["context_tokens"],
                "generation_tokens": values["generation_tokens"],
                "cpu_range_ms": round(float(annotation["dur"]) / 1000, 6),
                "gpu_range_ms": round(float(gpu_annotation["dur"]) / 1000, 6),
            }
        )

    kernel_events = [event for event in events if event.get("cat") == "kernel"]
    if not kernel_events:
        raise ValueError(f"{run_directory}: no CUDA kernel activity")
    family_time_us: Counter[str] = Counter()
    family_calls: Counter[str] = Counter()
    for event in kernel_events:
        family = _kernel_family(str(event["name"]))
        family_time_us[family] += float(event["dur"])
        family_calls[family] += 1

    classification_counts = Counter(row["classification"] for row in rows)
    return {
        "run_id": metadata["run_id"],
        "mode": expected_mode,
        "git_commit": metadata["git_commit"],
        "profile_step_start": delay,
        "profile_step_end": delay + len(rows) - 1,
        "profile_steps": len(rows),
        "classification_counts": dict(sorted(classification_counts.items())),
        "kernel_calls": len(kernel_events),
        "kernel_time_ms": round(sum(float(e["dur"]) for e in kernel_events) / 1000, 6),
        "kernel_family_calls": dict(sorted(family_calls.items())),
        "kernel_family_time_ms": {
            key: round(value / 1000, 6)
            for key, value in sorted(family_time_us.items())
        },
        "max_gpu_range_ms": max(row["gpu_range_ms"] for row in rows),
        "scheduler": scheduler,
        "step_rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    strict = analyze_profile(args.strict_run, "baseline")
    bounded = analyze_profile(args.bounded_run, "modified")
    if strict["git_commit"] != bounded["git_commit"]:
        raise ValueError("profiled runs used different commits")
    if (strict["profile_step_start"], strict["profile_step_end"]) != (
        bounded["profile_step_start"],
        bounded["profile_step_end"],
    ):
        raise ValueError("profiled runs used different Scheduler windows")

    summary = {
        "git_commit": strict["git_commit"],
        "strict": {key: value for key, value in strict.items() if key != "step_rows"},
        "bounded": {
            key: value for key, value in bounded.items() if key != "step_rows"
        },
        "bounded_vs_strict": {
            "kernel_time_ms_delta": round(
                bounded["kernel_time_ms"] - strict["kernel_time_ms"], 6
            ),
            "attention_time_ms_delta": round(
                bounded["kernel_family_time_ms"]["attention"]
                - strict["kernel_family_time_ms"]["attention"],
                6,
            ),
            "gemm_time_ms_delta": round(
                bounded["kernel_family_time_ms"].get("gemm", 0.0)
                - strict["kernel_family_time_ms"].get("gemm", 0.0),
                6,
            ),
        },
        "evidence_boundary": (
            "One bounded PyTorch-profiler window per mode; valid for observed "
            "step composition and kernel activity, not stable performance claims."
        ),
    }
    args.output_directory.mkdir(parents=True, exist_ok=False)
    _write_csv(
        args.output_directory / "day16_profile_steps.csv",
        strict["step_rows"] + bounded["step_rows"],
    )
    (args.output_directory / "day16_profile_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
