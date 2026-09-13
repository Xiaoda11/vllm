# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Sequence
from typing import Any


def _validate_worker_rank(worker_rank: int) -> int:
    rank = int(worker_rank)
    if rank < 0:
        raise ValueError("worker_rank must be non-negative")
    return rank


def make_model_runner_batch_event(
    *,
    step_id: int,
    worker_rank: int,
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
    rank = _validate_worker_rank(worker_rank)
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
        "worker_rank": rank,
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


def make_sampler_batch_shard_event(
    *,
    step_id: int,
    worker_rank: int,
    tp_rank: int,
    tp_size: int,
    global_request_ids: Sequence[str],
    global_persistent_rows: Sequence[int],
    local_request_ids: Sequence[str],
    local_persistent_rows: Sequence[int],
    num_logits_per_rank: Sequence[int],
    num_local_logits: int,
    max_num_reqs_per_rank: int,
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    """Build a CPU-only event for TP batch-sharded sampling ownership.

    v0.29 assigns request ownership by ``persistent_row % tp_size``. The event
    records the global replicated request layout and the local rank's shard so
    Scheduler/MRV2 traces can explain which rank actually sampled each request.
    GPU gather indices are intentionally excluded.
    """
    global_rank = _validate_worker_rank(worker_rank)
    rank = int(tp_rank)
    size = int(tp_size)
    if size <= 0 or not 0 <= rank < size:
        raise ValueError("TP rank/size must satisfy 0 <= rank < size")

    global_ids = list(global_request_ids)
    global_rows = [int(row) for row in global_persistent_rows]
    local_ids = list(local_request_ids)
    local_rows = [int(row) for row in local_persistent_rows]
    logits_per_rank = [int(count) for count in num_logits_per_rank]

    if len(global_ids) != len(global_rows):
        raise ValueError("global request IDs and persistent rows must align")
    if len(local_ids) != len(local_rows):
        raise ValueError("local request IDs and persistent rows must align")
    if len(logits_per_rank) != size:
        raise ValueError("num_logits_per_rank must have one entry per TP rank")

    owner_ranks = [row % size for row in global_rows]
    reqs_per_rank = [0] * size
    for owner in owner_ranks:
        reqs_per_rank[owner] += 1

    expected_local = [
        (req_id, row)
        for req_id, row, owner in zip(global_ids, global_rows, owner_ranks)
        if owner == rank
    ]
    if list(zip(local_ids, local_rows)) != expected_local:
        raise ValueError("local sampler shard does not match persistent-row ownership")
    if int(num_local_logits) != logits_per_rank[rank]:
        raise ValueError("num_local_logits does not match this rank's logit split")

    return {
        "schema_version": 1,
        "event": "sampler_batch_shard",
        "timestamp_ns": time.time_ns() if timestamp_ns is None else int(timestamp_ns),
        "step_id": int(step_id),
        "worker_rank": global_rank,
        "tp_rank": rank,
        "tp_size": size,
        "ownership": "persistent_row_mod_tp",
        "global_request_ids": global_ids,
        "global_persistent_rows": global_rows,
        "request_owner_ranks": owner_ranks,
        "local_request_ids": local_ids,
        "local_persistent_rows": local_rows,
        "num_reqs_per_rank": reqs_per_rank,
        "num_logits_per_rank": logits_per_rank,
        "num_local_logits": int(num_local_logits),
        "max_num_reqs_per_rank": int(max_num_reqs_per_rank),
    }
