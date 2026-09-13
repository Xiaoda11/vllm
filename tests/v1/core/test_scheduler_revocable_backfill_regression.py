# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _model_output(*req_ids: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1000 + i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_four_block_scheduler():
    # num_blocks includes the null block, so five total means four allocatable.
    return create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=32,
        max_model_len=32,
        num_blocks=5,
        block_size=4,
        enable_chunked_prefill=True,
    )


def _start_two_block_incumbent(scheduler):
    incumbent = create_requests(
        num_requests=1,
        num_tokens=7,
        max_tokens=8,
        block_size=4,
        req_ids=["incumbent"],
    )[0]
    scheduler.add_request(incumbent)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"incumbent": 7}
    scheduler.update_from_output(output, _model_output("incumbent"))
    return incumbent


def _add_heavy_and_backfill(scheduler):
    heavy = create_requests(
        num_requests=1,
        num_tokens=16,
        max_tokens=8,
        block_size=4,
        req_ids=["heavy"],
    )[0]
    backfill = create_requests(
        num_requests=1,
        num_tokens=3,
        max_tokens=8,
        block_size=4,
        req_ids=["backfill"],
    )[0]
    scheduler.add_request(heavy)
    scheduler.add_request(backfill)
    return heavy, backfill


def test_kv_blocked_head_reclaims_revocable_backfill() -> None:
    """A bypasser is reclaimable once it alone blocks the protected head."""
    scheduler = _create_four_block_scheduler()
    _start_two_block_incumbent(scheduler)
    _add_heavy_and_backfill(scheduler)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"incumbent": 1, "backfill": 3}
    assert "heavy" not in output.num_scheduled_tokens
    scheduler.update_from_output(output, _model_output("incumbent", "backfill"))

    # Three blocks become free. Reclaiming the tracked one-block backfill makes
    # all four allocatable blocks available to the protected head.
    scheduler.finish_requests("incumbent", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"heavy": 16}
    assert "backfill" in output.preempted_req_ids
    assert scheduler.requests["backfill"].status == RequestStatus.PREEMPTED
    assert "heavy" not in scheduler._kv_blocked_bypass_counts
    assert "heavy" not in scheduler._kv_blocked_backfill_ids


def test_untracked_running_request_is_not_reclaimed() -> None:
    """Ordinary incumbents must never become revocation victims by inference."""
    scheduler = _create_four_block_scheduler()
    _start_two_block_incumbent(scheduler)

    (heavy,) = create_requests(
        num_requests=1,
        num_tokens=16,
        max_tokens=8,
        block_size=4,
        req_ids=["heavy"],
    )
    scheduler.add_request(heavy)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"incumbent": 1}
    assert "heavy" not in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert [req.request_id for req in scheduler.running] == ["incumbent"]


def test_insufficient_backfill_is_not_reclaimed() -> None:
    """Do not preempt a backfill unless its KV is sufficient to unlock head."""
    scheduler = _create_four_block_scheduler()
    _start_two_block_incumbent(scheduler)
    _add_heavy_and_backfill(scheduler)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"incumbent": 1, "backfill": 3}
    scheduler.update_from_output(output, _model_output("incumbent", "backfill"))

    # Keep the incumbent alive. On this step it grows to three blocks, leaving
    # no free KV. Reclaiming the one-block backfill would still leave only one
    # block free, far short of the four required by heavy.
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"incumbent": 1, "backfill": 1}
    assert "heavy" not in output.num_scheduled_tokens
    assert not output.preempted_req_ids
    assert scheduler.requests["backfill"].status == RequestStatus.RUNNING


def test_finished_backfill_is_removed_from_tracking() -> None:
    """Tracking must not retain a bypasser after it leaves the scheduler."""
    scheduler = _create_four_block_scheduler()
    _start_two_block_incumbent(scheduler)
    _add_heavy_and_backfill(scheduler)

    output = scheduler.schedule()
    scheduler.update_from_output(output, _model_output("incumbent", "backfill"))

    assert scheduler._kv_blocked_backfill_ids["heavy"] == ["backfill"]
    scheduler.finish_requests("backfill", RequestStatus.FINISHED_ABORTED)

    assert "backfill" not in scheduler._kv_blocked_backfill_ids.get("heavy", [])
