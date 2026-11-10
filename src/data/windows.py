"""Select the common high-quality navigation interval."""

import numpy as np
from src.data.geometry import BASE_HZ


def paper_window_start(navigation, duration):
    """Select a CATS-CF paper window inside the common navigation interval."""
    duration = float(duration)
    if (
        duration <= 0
        or not np.isfinite(duration)
        or not (duration * BASE_HZ).is_integer()
    ):
        raise ValueError(
            "Window length must be a positive whole number of native samples"
        )
    parent_duration = duration
    cars = [
        navigation.dropna(subset=[f"{role}_ins_status"])
        for role in ("leader", "follower")
    ]
    start = max(car.t_rel_s.iloc[0] for car in cars)
    end = min(car.t_rel_s.iloc[-1] for car in cars)
    if end - start < parent_duration - 1e-8:
        raise ValueError(
            f"Common interval is too short for a {parent_duration:g} s window"
        )
    candidates = start + np.arange(
        0.0, end - start - parent_duration + 0.5 / BASE_HZ, 1 / BASE_HZ
    )
    severe = np.zeros(len(candidates), dtype=int)
    degraded = np.zeros_like(severe)
    for role, car in zip(("leader", "follower"), cars):
        t = car.t_rel_s.to_numpy()
        left, right = (
            np.searchsorted(t, candidates),
            np.searchsorted(t, candidates + parent_duration),
        )
        ins_bad = car[f"{role}_ins_status"].ne(3).to_numpy(dtype=int)
        pos = car[f"{role}_position_type"]
        for scores, flags in (
            (severe, ins_bad + pos.isin([0, 53])),
            (degraded, ins_bad + pos.ne(56)),
        ):
            prefix = np.r_[0, np.cumsum(flags)]
            scores += prefix[right] - prefix[left]
    best = candidates[np.lexsort((candidates, degraded, severe))[0]]
    return best
