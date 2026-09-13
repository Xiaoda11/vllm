# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Sequence
from typing import Any


def make_model_runner_batch_event(
    *,
    step_id: int,
    request_ids: Sequence[str],
    persistent_rows: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    scheduler_total_num_scheduled_tokens: int,
    runner_num_tokens: int,
    model_num_tokens_after_padding: int,
    num_reqs_after_padding: int,
    has_prefill: bool,
    cudagraph_mode: str,
    adaptive_verification_active: bool,
    batch_sharded_sampling_enabled: bool,
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Build one CPU-only MRV2 batch trace event.

    ``num_scheduled_tokens`` and ``scheduler_total_num_scheduled_tokens`` are
    Scheduler-logical quantities. ``runner_num_tokens`` is the unpadded token
    count the runner intends to execute after any CPU-visible trimming, while
    ``model_num_tokens_after_padding`` includes CUDA-graph padding.

    The caller must pass only CPU/Python values. This helper deliberately does
    not accept torch tensors so tracing cannot accidentally introduce a D2H
    transfer or synchronization.
    """
    req_ids = list(request_ids)
    rows = [int(row) for row in persistent_rows]
    scheduled = [int(count) for count in num_scheduled_tokens]

    if len(req_ids) != len(rows) or len(req_ids) != len(scheduled):
        raise ValueError(
            "request_ids, persistent_rows and num_scheduled_tokens must have "
            "identical lengths"
        )

    scheduler_total = int(scheduler_total_num_scheduled_tokens)
    runner_tokens = int(runner_num_tokens)
    model_tokens = int(model_num_tokens_after_padding)
    padded_reqs = int(num_reqs_after_padding)
    if runner_tokens < 0 or model_tokens < runner_tokens:
        raise ValueError("model token counts must satisfy 0 <= runner <= padded")
    if padded_reqs < len(req_ids):
        raise ValueError("num_reqs_after_padding cannot be smaller than request count")

    return {
        "schema_version": 1,
        "event": "model_runner_batch",
        "timestamp_ns": time.time_ns() if timestamp_ns is None else int(timestamp_ns),
        "step_id": int(step_id),
        # Backward-compatible v0.26 fields.
        "request_ids": req_ids,
        "persistent_rows": rows,
        "num_scheduled_tokens": scheduled,
        "request_id_to_persistent_row": dict(zip(req_ids, rows)),
        # v0.29 distinguishes scheduler budget from actual runner/model shapes.
        "token_counts": {
            "scheduler_logical": scheduler_total,
            "runner_effective_unpadded": runner_tokens,
            "model_input_after_padding": model_tokens,
            "scheduler_to_runner_delta": scheduler_total - runner_tokens,
            "graph_padding": model_tokens - runner_tokens,
        },
        "request_counts": {
            "logical": len(req_ids),
            "model_input_after_padding": padded_reqs,
        },
        "has_prefill": bool(has_prefill),
        "cudagraph_mode": cudagraph_mode,
        "adaptive_verification_active": bool(adaptive_verification_active),
        "batch_sharded_sampling_enabled": bool(batch_sharded_sampling_enabled),
    }
