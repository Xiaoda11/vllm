#!/usr/bin/env python3
"""Flatten Scheduler Trace Lab JSONL into one CSV row per request and step."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "step_id",
    "request_id",
    "scheduled_request_ids",
    "running_before",
    "waiting_before",
    "running_after",
    "waiting_after",
    "status_before",
    "status_after",
    "prompt_tokens",
    "output_tokens_before",
    "computed_tokens_before",
    "computed_tokens_after_schedule",
    "in_flight_tokens_before",
    "in_flight_tokens_after_schedule",
    "processed_tokens_before",
    "processed_tokens_after_schedule",
    "num_scheduled_tokens",
    "prefix_cached_tokens",
    "token_budget_initial",
    "token_budget_scheduled",
    "token_budget_remaining",
    "kv_usage_before",
    "kv_usage_after",
    "num_free_blocks_before",
    "num_free_blocks_after",
    "kv_block_size",
    "prefix_cached_blocks",
    "block_table_blocks_before",
    "block_table_blocks_after",
    "block_table_blocks_added",
    "block_table_blocks_removed",
    "allocated_block_ids",
    "freed_block_ids",
    "allocation_failed",
    "allocation_failure_phase",
    "preempted_computed_tokens",
    "preempted",
    "finished",
    "mrv2_execution_order",
    "persistent_row",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument("--model-runner-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("schema_version") != 1:
                raise ValueError(f"{path}:{line_number}: unsupported schema version")
            records.append(record)
    return records


def read_runner_steps(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    records = read_jsonl(path)
    return {
        int(record["step_id"]): record
        for record in records
        if record.get("event") == "model_runner_batch"
    }


def _join(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def _state_value(
    request: dict[str, Any],
    state_name: str,
    field: str,
    default: Any = "",
) -> Any:
    state = request.get(state_name)
    return state.get(field, default) if state is not None else default


def _block_count(groups: list[list[int]]) -> int:
    return sum(len(group) for group in groups)


def flatten_trace(
    scheduler_records: list[dict[str, Any]],
    runner_steps: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for step in scheduler_records:
        if step.get("event") != "scheduler_step":
            continue
        step_id = int(step["step_id"])
        runner_step = runner_steps.get(step_id, {})
        runner_order = runner_step.get("request_ids", [])
        row_by_request = runner_step.get("request_id_to_persistent_row", {})
        requests = step["requests"] or [{"request_id": ""}]
        for request in requests:
            request_id = request["request_id"]
            failures = [
                failure
                for failure in step.get("allocation_failures", [])
                if failure["request_id"] == request_id
            ]
            preempted_computed_tokens = sum(
                failure.get("preempted_num_computed_tokens") or 0
                for failure in step.get("allocation_failures", [])
                if failure.get("preempted_request_id") == request_id
            )
            before = request.get("before")
            after = request.get("after")
            blocks_before = _state_value(request, "before", "block_ids", [])
            blocks_after = _state_value(request, "after", "block_ids", [])
            allocated_block_ids = request["allocated_block_ids"]
            freed_block_ids = request["freed_block_ids"]
            block_size = step["kv_cache"].get("block_size")
            prefix_cached_tokens = request["prefix_cached_tokens"]
            rows.append(
                {
                    "step_id": step_id,
                    "request_id": request_id,
                    "scheduled_request_ids": _join(step["scheduled_request_ids"]),
                    "running_before": _join(step["queues"]["running_before"]),
                    "waiting_before": _join(step["queues"]["waiting_before"]),
                    "running_after": _join(step["queues"]["running_after"]),
                    "waiting_after": _join(step["queues"]["waiting_after"]),
                    "status_before": (before["status"] if before is not None else ""),
                    "status_after": (after["status"] if after is not None else ""),
                    "prompt_tokens": _state_value(
                        request,
                        "after",
                        "num_prompt_tokens",
                        _state_value(request, "before", "num_prompt_tokens"),
                    ),
                    "output_tokens_before": _state_value(
                        request, "before", "num_output_tokens"
                    ),
                    "computed_tokens_before": _state_value(
                        request, "before", "num_computed_tokens"
                    ),
                    "computed_tokens_after_schedule": _state_value(
                        request, "after", "num_computed_tokens"
                    ),
                    "in_flight_tokens_before": _state_value(
                        request, "before", "num_in_flight_tokens"
                    ),
                    "in_flight_tokens_after_schedule": _state_value(
                        request, "after", "num_in_flight_tokens"
                    ),
                    "processed_tokens_before": _state_value(
                        request, "before", "num_processed_tokens"
                    ),
                    "processed_tokens_after_schedule": _state_value(
                        request, "after", "num_processed_tokens"
                    ),
                    "num_scheduled_tokens": request["num_scheduled_tokens"],
                    "prefix_cached_tokens": prefix_cached_tokens,
                    "token_budget_initial": step["token_budget"]["initial"],
                    "token_budget_scheduled": step["token_budget"]["scheduled"],
                    "token_budget_remaining": step["token_budget"]["remaining"],
                    "kv_usage_before": step["kv_cache"]["usage_before"],
                    "kv_usage_after": step["kv_cache"]["usage_after"],
                    "num_free_blocks_before": step["kv_cache"][
                        "num_free_blocks_before"
                    ],
                    "num_free_blocks_after": step["kv_cache"]["num_free_blocks_after"],
                    "kv_block_size": block_size or "",
                    "prefix_cached_blocks": (
                        prefix_cached_tokens // block_size if block_size else ""
                    ),
                    "block_table_blocks_before": _block_count(blocks_before),
                    "block_table_blocks_after": _block_count(blocks_after),
                    "block_table_blocks_added": _block_count(allocated_block_ids),
                    "block_table_blocks_removed": _block_count(freed_block_ids),
                    "allocated_block_ids": json.dumps(
                        allocated_block_ids,
                        separators=(",", ":"),
                    ),
                    "freed_block_ids": json.dumps(
                        freed_block_ids,
                        separators=(",", ":"),
                    ),
                    "allocation_failed": bool(failures),
                    "allocation_failure_phase": _join(
                        [failure["phase"] for failure in failures]
                    ),
                    "preempted_computed_tokens": preempted_computed_tokens,
                    "preempted": request_id in step["preempted_request_ids"],
                    "finished": request_id in step["finished_request_ids"],
                    "mrv2_execution_order": _join(runner_order),
                    "persistent_row": row_by_request.get(request_id, ""),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scheduler_records = read_jsonl(args.scheduler_trace)
    runner_steps = read_runner_steps(args.model_runner_trace)
    rows = flatten_trace(scheduler_records, runner_steps)
    write_csv(args.output, rows)
    print(
        json.dumps(
            {
                "scheduler_steps": len(scheduler_records),
                "csv_rows": len(rows),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
