# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-Python construction helpers for Scheduler Trace Lab events.

Keep event construction separate from Scheduler integration so schema semantics
can be validated on CPU-only CI without importing vLLM or torch.
"""

import time
from typing import Any


def block_id_difference(
    left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    """Return block ids present in ``left`` but not ``right``, per cache group."""
    num_groups = max(len(left), len(right))
    differences: list[list[int]] = []
    for group_index in range(num_groups):
        left_group = left[group_index] if group_index < len(left) else []
        right_group = set(right[group_index]) if group_index < len(right) else set()
        differences.append(
            [block_id for block_id in left_group if block_id not in right_group]
        )
    return differences


def make_scheduler_step_event(
    *,
    step_id: int,
    before: dict[str, Any],
    after: dict[str, Any],
    num_scheduled_tokens: dict[str, int],
    total_num_scheduled_tokens: int,
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
    token_budget_initial: int,
    token_budget_remaining: int,
    input_budget_initial: int,
    input_budget_remaining: int,
    block_size: int,
    prefix_cached_tokens: dict[str, int] | None = None,
    allocation_failures: list[dict[str, Any]] | None = None,
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Build one version-1 scheduler-step event from CPU-side state snapshots."""
    prefix_cached_tokens = prefix_cached_tokens or {}
    allocation_failures = allocation_failures or []
    scheduled_ids = list(num_scheduled_tokens)

    request_ids = list(
        dict.fromkeys(
            [
                *before["running"],
                *before["waiting"],
                *before.get("skipped_waiting", []),
                *scheduled_ids,
                *after["running"],
                *after["waiting"],
                *after.get("skipped_waiting", []),
                *sorted(finished_req_ids),
            ]
        )
    )

    request_events: list[dict[str, Any]] = []
    for request_id in request_ids:
        before_state = before["requests"].get(request_id)
        after_state = after["requests"].get(request_id)
        before_blocks = before_state["block_ids"] if before_state else []
        after_blocks = after_state["block_ids"] if after_state else []
        request_events.append(
            {
                "request_id": request_id,
                "before": before_state,
                "after": after_state,
                "num_scheduled_tokens": num_scheduled_tokens.get(request_id, 0),
                "prefix_cached_tokens": prefix_cached_tokens.get(request_id, 0),
                "allocated_block_ids": block_id_difference(after_blocks, before_blocks),
                "freed_block_ids": block_id_difference(before_blocks, after_blocks),
            }
        )

    return {
        "schema_version": 1,
        "event": "scheduler_step",
        "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
        "step_id": step_id,
        "token_budget": {
            "initial": token_budget_initial,
            "scheduled": total_num_scheduled_tokens,
            "remaining": token_budget_remaining,
        },
        "input_budget": {
            "initial": input_budget_initial,
            "remaining": input_budget_remaining,
        },
        "queues": {
            "running_before": before["running"],
            "waiting_before": before["waiting"],
            "skipped_waiting_before": before.get("skipped_waiting", []),
            "running_after": after["running"],
            "waiting_after": after["waiting"],
            "skipped_waiting_after": after.get("skipped_waiting", []),
        },
        "kv_cache": {
            "block_size": block_size,
            "usage_before": before["kv_cache_usage"],
            "usage_after": after["kv_cache_usage"],
            "num_free_blocks_before": before["num_free_blocks"],
            "num_free_blocks_after": after["num_free_blocks"],
        },
        "requests": request_events,
        "allocation_failures": allocation_failures,
        "scheduled_request_ids": scheduled_ids,
        "preempted_request_ids": sorted(preempted_req_ids or ()),
        "finished_request_ids": sorted(finished_req_ids),
    }
