# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.revocable_backfill import RevocableBackfillState


def test_only_one_active_backfill_per_protected_head():
    state = RevocableBackfillState()
    state.protect("heavy")

    assert state.can_admit_backfill("light-1")
    state.admit_backfill("light-1")

    assert not state.can_admit_backfill("light-2")


def test_reclaim_only_when_backfill_blocks_head_admission():
    state = RevocableBackfillState()
    state.protect("heavy")
    state.admit_backfill("light")

    assert state.should_reclaim(
        free_blocks=3,
        protected_required_blocks=4,
        backfill_held_blocks=1,
    )

    assert not state.should_reclaim(
        free_blocks=4,
        protected_required_blocks=4,
        backfill_held_blocks=1,
    )

    assert not state.should_reclaim(
        free_blocks=2,
        protected_required_blocks=4,
        backfill_held_blocks=1,
    )


def test_reclaimed_backfill_cannot_bypass_same_head_again():
    state = RevocableBackfillState()
    state.protect("heavy")
    state.admit_backfill("light")

    assert state.mark_reclaimed() == ("light", "heavy")
    assert not state.can_admit_backfill("light")
    assert state.can_admit_backfill("other-light")


def test_reclaimed_request_can_backfill_for_different_head():
    state = RevocableBackfillState()
    state.protect("heavy-1")
    state.admit_backfill("light")
    state.mark_reclaimed()

    state.protect("heavy-2")
    assert state.can_admit_backfill("light")


def test_backfill_release_allows_another_candidate():
    state = RevocableBackfillState()
    state.protect("heavy")
    state.admit_backfill("light-1")

    state.release_backfill("light-1")
    assert state.can_admit_backfill("light-2")


def test_clearing_head_clears_active_relationship():
    state = RevocableBackfillState()
    state.protect("heavy")
    state.admit_backfill("light")

    state.clear_protected_head("heavy")
    assert state.protected_head_id is None
    assert state.active_backfill_id is None
    assert not state.can_admit_backfill("light-2")


def test_invalid_backfill_admission_is_rejected():
    state = RevocableBackfillState()

    with pytest.raises(ValueError, match="not eligible"):
        state.admit_backfill("light")
