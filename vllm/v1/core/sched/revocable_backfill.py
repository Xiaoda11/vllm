# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field


@dataclass(slots=True)
class RevocableBackfillState:
    """Scheduler-local state for one protected FCFS head.

    The policy allows at most one younger request to backfill while the
    protected head is blocked on local KV allocation. A reclaimed backfill
    cannot bypass the same protected head again.
    """

    protected_head_id: str | None = None
    active_backfill_id: str | None = None
    revoked_pairs: set[tuple[str, str]] = field(default_factory=set)

    def protect(self, head_id: str) -> None:
        if self.protected_head_id == head_id:
            return
        self.protected_head_id = head_id
        self.active_backfill_id = None

    def clear_protected_head(self, head_id: str) -> None:
        if self.protected_head_id != head_id:
            return
        self.protected_head_id = None
        self.active_backfill_id = None

    def can_admit_backfill(self, request_id: str) -> bool:
        head_id = self.protected_head_id
        if head_id is None or self.active_backfill_id is not None:
            return False
        return (request_id, head_id) not in self.revoked_pairs

    def admit_backfill(self, request_id: str) -> None:
        if not self.can_admit_backfill(request_id):
            raise ValueError("request is not eligible for revocable backfill")
        self.active_backfill_id = request_id

    def release_backfill(self, request_id: str) -> None:
        if self.active_backfill_id == request_id:
            self.active_backfill_id = None

    def should_reclaim(
        self,
        *,
        free_blocks: int,
        protected_required_blocks: int,
        backfill_held_blocks: int,
    ) -> bool:
        """Return whether reclaiming the active backfill unlocks the head."""
        if self.protected_head_id is None or self.active_backfill_id is None:
            return False
        if protected_required_blocks <= free_blocks:
            return False
        return protected_required_blocks <= free_blocks + backfill_held_blocks

    def mark_reclaimed(self) -> tuple[str, str]:
        head_id = self.protected_head_id
        backfill_id = self.active_backfill_id
        if head_id is None or backfill_id is None:
            raise ValueError("no active protected-head/backfill pair to reclaim")
        pair = (backfill_id, head_id)
        self.revoked_pairs.add(pair)
        self.active_backfill_id = None
        return pair
