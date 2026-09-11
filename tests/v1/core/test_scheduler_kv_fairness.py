# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


def _model_output(*req_ids: str) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[1000 + i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_waiting_kv_blocked_request_does_not_block_lighter_request():
    # BlockPool reserves block 0 as the null block, so num_blocks=2 gives
    # exactly one allocatable KV block for this test.
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        num_blocks=2,
        block_size=4,
        enable_chunked_prefill=True,
    )

    heavy = create_requests(
        num_requests=1, num_tokens=9, block_size=4, req_ids=["heavy"]
    )[0]
    light = create_requests(
        num_requests=1, num_tokens=4, block_size=4, req_ids=["light"]
    )[0]

    scheduler.add_request(heavy)
    scheduler.add_request(light)

    output = scheduler.schedule()
    scheduled_ids = [req.req_id for req in output.scheduled_new_reqs]
    assert scheduled_ids == ["light"]


def test_retrying_blocked_head_first_does_not_prevent_extra_waiting():
    """A retried FCFS head can still wait behind KV retained by backfill.

    There are four allocatable blocks (five total minus the null block).
    The incumbent holds two blocks, so the four-block heavy request cannot
    enter. The one-block light request can bypass it. After the incumbent is
    removed, the heavy request is retried first, but the light request still
    retains one block, leaving only three free. Once the light request is
    removed, the heavy request becomes immediately schedulable.

    This is a deterministic fairness-boundary test. It does not assert a
    particular starvation-prevention policy; it only demonstrates that
    retrying a skipped head before younger waiting requests is not by itself a
    guarantee that the head incurs no additional KV-admission delay.
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
    light = create_requests(
        num_requests=1,
        num_tokens=3,
        max_tokens=8,
        block_size=4,
        req_ids=["light"],
    )[0]
    scheduler.add_request(heavy)
    scheduler.add_request(light)

    # Two blocks are occupied by the incumbent. Heavy needs all four usable
    # blocks, so it is skipped; light needs one and is admitted behind it.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"incumbent": 1, "light": 3}
    assert "heavy" not in output.num_scheduled_tokens
    assert [request.request_id for request in scheduler.waiting] == ["heavy"]
    scheduler.update_from_output(output, _model_output("incumbent", "light"))

    # Free the older incumbent. Heavy is now the first waiting request again,
    # but the bypassed light request retains one block. Only three of four
    # usable blocks are free, so heavy still cannot enter.
    scheduler.finish_requests("incumbent", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"light": 1}
    assert "heavy" not in output.num_scheduled_tokens
    assert [request.request_id for request in scheduler.waiting] == ["heavy"]
    scheduler.update_from_output(output, _model_output("light"))

    # The heavy request itself is feasible: once the bypassed request releases
    # its block, heavy can consume the full four-block capacity immediately.
    scheduler.finish_requests("light", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"heavy": 16}
