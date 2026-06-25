from __future__ import annotations

import math


def waitlist_expansion_threshold(capacity: int, threshold_ratio: float = 0.50) -> int:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if threshold_ratio < 0:
        raise ValueError("threshold_ratio must be nonnegative")
    return math.ceil(capacity * threshold_ratio)


def waitlist_triggers_expansion(
    waitlist_count: int,
    capacity: int,
    threshold_ratio: float = 0.50,
) -> bool:
    if waitlist_count < 0:
        raise ValueError("waitlist_count must be nonnegative")
    return waitlist_count >= waitlist_expansion_threshold(capacity, threshold_ratio)
