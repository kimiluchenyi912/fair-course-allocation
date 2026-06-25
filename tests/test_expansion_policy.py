from __future__ import annotations

import pytest

from src.expansion_policy import waitlist_expansion_threshold, waitlist_triggers_expansion


@pytest.mark.parametrize(
    ("capacity", "waitlist_before", "waitlist_at_threshold", "threshold"),
    [
        (40, 19, 20, 20),
        (25, 12, 13, 13),
        (50, 24, 25, 25),
    ],
)
def test_waitlist_expansion_uses_ceiling_of_50_percent(
    capacity: int,
    waitlist_before: int,
    waitlist_at_threshold: int,
    threshold: int,
) -> None:
    assert waitlist_expansion_threshold(capacity) == threshold
    assert not waitlist_triggers_expansion(waitlist_before, capacity)
    assert waitlist_triggers_expansion(waitlist_at_threshold, capacity)


def test_waitlist_expansion_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        waitlist_expansion_threshold(0)
    with pytest.raises(ValueError):
        waitlist_triggers_expansion(-1, 40)
