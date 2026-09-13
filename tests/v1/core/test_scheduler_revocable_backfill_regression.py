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


def test_kv_blocked_head_reclaims_revocable_backfill() -> None:
    """A bypasser should be reclaimable once it alone blocks the protected head.

    Four KV blocks are usable (block 0 is the null block). An incumbent first
    holds two blocks. While the older four-block request is KV-blocked, a younger
    one-block request backfills otherwise idle capacity. After the incumbent
    exits, exactly three blocks are free; reclaiming the one-block backfill is
    therefore sufficient to make the older request admissible.

    A bounded *future admission* count does not protect this opportunity because
    the bypasser is already running. Revocable-backfill semantics should preempt
    that bypasser and admit the protected head in this step.
    """
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=32,
        max_model_len=32,
        num_blocks=5,
        block_size=4,
        enable_chunked_prefill=True,
    )

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

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"incumbent": 1, "backfill": 3}
    assert "heavy" not in output.num_scheduled_tokens
    scheduler.update_from_output(output, _model_output("incumbent", "backfill"))

    # The incumbent releases two blocks. Three blocks are now free while the
    # already-admitted backfill retains one. Reclaiming just that backfill would
    # make all four usable blocks available to the protected head.
    scheduler.finish_requests("incumbent", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()

    assert "heavy" in output.num_scheduled_tokens
    assert "backfill" in output.preempted_req_ids
