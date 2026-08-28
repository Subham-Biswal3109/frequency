"""
Transparent, documented spectrum allocation algorithm for the Spectrum
Simulation module. This is an EDUCATIONAL/ENGINEERING DEMONSTRATION only —
it is not, and must never be described as, the allocation mechanism used
by TRAI, DoT, or any telecom operator.

Allocation algorithm (see README section "Spectrum Simulation" for the
narrated version of this same logic):

  1. Only channels whose reconciled `state` is AVAILABLE (i.e. the RF
     sensing pipeline found no signal AND, when the ML model is loaded,
     the model also predicts availability) are eligible.
  2. Eligible channels are grouped into maximal contiguous runs.
  3. Within each run, every contiguous window whose combined bandwidth
     satisfies the requested bandwidth is a *candidate*.
  4. Each candidate is scored from three ingredients:
       - average ML availability probability (primary signal, weight 0.60)
       - average RF SNR, min-max scaled to 0..30 dB (weight 0.25)
       - isolation: how many free channels separate the candidate from the
         nearest OCCUPIED channel, normalized by grid size (weight 0.15)
     These weights are an explicit, adjustable design choice documented
     here — not a property of the ML model itself.
  5. Candidates are ranked by score; the highest-scoring candidate is
     selected. If no candidate satisfies the requested bandwidth, no
     allocation is made and a clear message is returned instead of
     forcing a result.
"""

import copy
from typing import List, Dict, Optional, Tuple

SCORE_WEIGHT_ML_PROBABILITY = 0.60
SCORE_WEIGHT_SNR = 0.25
SCORE_WEIGHT_ISOLATION = 0.15
SNR_NORMALIZATION_RANGE_DB = (0.0, 30.0)


def _normalize_snr(snr_db: float) -> float:
    lo, hi = SNR_NORMALIZATION_RANGE_DB
    return max(0.0, min(1.0, (snr_db - lo) / (hi - lo)))


def _contiguous_runs(available_channels: List[Dict]) -> List[List[Dict]]:
    """Groups AVAILABLE channels (already sorted by channel_id) into maximal
    runs where consecutive channels are frequency-adjacent."""
    if not available_channels:
        return []
    runs: List[List[Dict]] = [[available_channels[0]]]
    for ch in available_channels[1:]:
        last = runs[-1][-1]
        if ch["channel_id"] == last["channel_id"] + 1:
            runs[-1].append(ch)
        else:
            runs.append([ch])
    return runs


def _isolation_score(channels_by_id: Dict[int, Dict], window: List[Dict], total_channels: int) -> float:
    first_id = window[0]["channel_id"]
    last_id = window[-1]["channel_id"]

    gap_before = 0
    cid = first_id - 1
    while cid in channels_by_id and channels_by_id[cid]["rf_state"] != "OCCUPIED":
        gap_before += 1
        cid -= 1

    gap_after = 0
    cid = last_id + 1
    while cid in channels_by_id and channels_by_id[cid]["rf_state"] != "OCCUPIED":
        gap_after += 1
        cid += 1

    isolation = min(gap_before, gap_after) if (first_id - 1 in channels_by_id or last_id + 1 in channels_by_id) else total_channels
    return max(0.0, min(1.0, isolation / max(total_channels, 1)))


def find_candidates(channels: List[Dict], requested_bandwidth_mhz: float) -> List[Dict]:
    """Builds every valid contiguous candidate window that can satisfy the
    requested bandwidth, with its transparent scoring breakdown."""
    channels_by_id = {c["channel_id"]: c for c in channels}
    eligible = [
        c for c in channels
        if c["state"] == "AVAILABLE"
    ]
    eligible.sort(key=lambda c: c["channel_id"])

    candidates: List[Dict] = []
    for run in _contiguous_runs(eligible):
        n = len(run)
        for start_idx in range(n):
            total_bw = 0.0
            for end_idx in range(start_idx, n):
                window = run[start_idx:end_idx + 1]
                total_bw = sum(c["bandwidth_mhz"] for c in window)
                if total_bw >= requested_bandwidth_mhz:
                    ml_probs = [c.get("ml_probability") for c in window]
                    avg_ml_probability = (
                        sum(p for p in ml_probs if p is not None) / len(ml_probs)
                        if all(p is not None for p in ml_probs) and ml_probs
                        else None
                    )
                    avg_snr = sum(c["rf_snr_db"] for c in window) / len(window)
                    isolation = _isolation_score(channels_by_id, window, len(channels))

                    ml_component = avg_ml_probability if avg_ml_probability is not None else 0.5
                    score = (
                        SCORE_WEIGHT_ML_PROBABILITY * ml_component
                        + SCORE_WEIGHT_SNR * _normalize_snr(avg_snr)
                        + SCORE_WEIGHT_ISOLATION * isolation
                    )

                    candidates.append({
                        "channel_ids": [c["channel_id"] for c in window],
                        "start_mhz": window[0]["start_mhz"],
                        "end_mhz": window[-1]["end_mhz"],
                        "total_bandwidth_mhz": round(total_bw, 3),
                        "avg_snr_db": round(avg_snr, 2),
                        "avg_ml_probability": round(avg_ml_probability, 4) if avg_ml_probability is not None else None,
                        "isolation_score": round(isolation, 3),
                        "score": round(score, 4),
                    })
                    break  # smallest window at this start_idx that satisfies bandwidth; move on
    return candidates


def rank_candidates(candidates: List[Dict]) -> List[Dict]:
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked


def allocate(channels: List[Dict], requested_bandwidth_mhz: float) -> Tuple[Optional[Dict], List[Dict], str]:
    """
    Returns (selected_candidate_or_None, ranked_candidates, message).
    Never forces an allocation when no valid candidate exists.
    """
    candidates = find_candidates(channels, requested_bandwidth_mhz)
    if not candidates:
        return None, [], (
            f"No suitable channel found for the requested {requested_bandwidth_mhz} MHz bandwidth. "
            "Simulated allocation could not be completed with the current spectrum configuration."
        )

    ranked = rank_candidates(candidates)
    selected = ranked[0]
    reason = (
        f"Selected because it satisfies the requested {requested_bandwidth_mhz} MHz bandwidth and has "
        f"the highest availability score ({selected['score']}) among {len(ranked)} valid candidate "
        "window(s). Simulated allocation — not a guarantee of interference-free operation."
    )
    return selected, ranked, reason


def apply_allocation(channels: List[Dict], selected_candidate: Dict) -> List[Dict]:
    """Returns a NEW channel list with the selected candidate's channels marked ALLOCATED."""
    updated = copy.deepcopy(channels)
    allocated_ids = set(selected_candidate["channel_ids"])
    for c in updated:
        if c["channel_id"] in allocated_ids:
            c["state"] = "ALLOCATED"
    return updated


def resource_utilization(channels: List[Dict]) -> Dict:
    """Section 17 style summary — recalculated dynamically from actual channel states."""
    total_mhz = sum(c["bandwidth_mhz"] for c in channels)
    occupied_mhz = sum(c["bandwidth_mhz"] for c in channels if c["rf_state"] == "OCCUPIED")
    allocated_mhz = sum(c["bandwidth_mhz"] for c in channels if c["state"] == "ALLOCATED")
    available_mhz = sum(
        c["bandwidth_mhz"] for c in channels
        if c["state"] == "AVAILABLE"
    )
    return {
        "total_mhz": round(total_mhz, 3),
        "occupied_mhz": round(occupied_mhz, 3),
        "available_mhz": round(available_mhz, 3),
        "allocated_mhz": round(allocated_mhz, 3),
    }


def allocate_multi_user(channels: List[Dict], users: List[Dict]) -> Dict:
    """
    users: [{"user_id": str, "requested_bandwidth_mhz": float}, ...]
    Allocates sequentially; each user only sees spectrum left over by
    previous users in the list. Overlapping allocations are impossible by
    construction since allocated channels are removed from eligibility.
    """
    working_channels = copy.deepcopy(channels)
    user_results = []
    utilization_timeline = [resource_utilization(working_channels)]

    for user in users:
        selected, ranked, message = allocate(working_channels, user["requested_bandwidth_mhz"])
        if selected:
            working_channels = apply_allocation(working_channels, selected)
        user_results.append({
            "user_id": user.get("user_id", f"user_{len(user_results) + 1}"),
            "requested_bandwidth_mhz": user["requested_bandwidth_mhz"],
            "success": selected is not None,
            "selected": selected,
            "top_candidates": ranked[:5],
            "message": message,
        })
        utilization_timeline.append(resource_utilization(working_channels))

    return {
        "user_results": user_results,
        "final_channels": working_channels,
        "utilization_timeline": utilization_timeline,
    }
