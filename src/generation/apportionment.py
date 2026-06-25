from __future__ import annotations

import math


def largest_remainder(total: int, shares: dict[int, float]) -> dict[int, int]:
    """Allocate an integer total according to fractional shares."""
    raw = {key: total * share for key, share in shares.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts
