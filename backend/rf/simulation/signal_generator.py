"""
Seeded random assignment of "existing users" (occupying signals) onto a
channel grid, using physically plausible power ranges.

This module ONLY decides *which* channels have a transmitting signal and at
what nominal power — it does not itself decide the final OCCUPIED/AVAILABLE
state. That decision is made downstream by actually running the existing
FFT/PSD + peak-detection pipeline (see spectrum_simulator.py) against a
signal generated from these parameters, so the "RF sensing" result is a
genuine detection outcome rather than a hard-coded label.
"""

import random
from typing import List, Dict, Optional

# Physically plausible ranges, matching the project's existing conventions
# in backend/rf/rf_source.py and the ranges used by the ML training data.
SIGNAL_POWER_RANGE_DBM = (-90.0, -60.0)


def assign_existing_signals(
    channels: List[Dict],
    num_existing_users: int,
    seed: Optional[int] = None,
) -> Dict[int, float]:
    """
    Randomly selects `num_existing_users` distinct channels (or fewer, if the
    grid is too small) to carry an existing signal, and assigns each a
    signal power in dBm drawn from SIGNAL_POWER_RANGE_DBM.

    Returns a dict mapping channel_id -> signal_power_dbm for occupied
    channels only. Channels not present in the dict have no existing
    carrier (thermal noise only).
    """
    rng = random.Random(seed)

    num_existing_users = max(0, int(num_existing_users))
    num_existing_users = min(num_existing_users, len(channels))

    occupied_channel_ids = rng.sample(
        [c["channel_id"] for c in channels], k=num_existing_users
    )

    assignments: Dict[int, float] = {}
    for channel_id in occupied_channel_ids:
        power_dbm = rng.uniform(*SIGNAL_POWER_RANGE_DBM)
        assignments[channel_id] = round(power_dbm, 2)

    return assignments
