"""
Channel grid generation for the Spectrum Simulation module.

Divides a configured [start_frequency_mhz, end_frequency_mhz) band into
equal-width channels of `channel_bandwidth_mhz`. This is pure arithmetic
(no randomness) so it is identical for a given configuration regardless
of simulation seed.
"""

from typing import List, Dict


def generate_channel_grid(
    start_frequency_mhz: float,
    end_frequency_mhz: float,
    channel_bandwidth_mhz: float,
) -> List[Dict]:
    """
    Returns a list of non-overlapping channel dicts spanning the requested
    frequency range:
        {
            "channel_id": int (1-indexed),
            "start_mhz": float,
            "end_mhz": float,
            "center_mhz": float,
            "bandwidth_mhz": float,
        }

    Any remainder smaller than a full channel width at the end of the range
    is dropped (we only emit full-width channels), matching how a real
    channelization scheme would behave.
    """
    if end_frequency_mhz <= start_frequency_mhz:
        raise ValueError("end_frequency_mhz must be greater than start_frequency_mhz")
    if channel_bandwidth_mhz <= 0:
        raise ValueError("channel_bandwidth_mhz must be greater than 0")

    total_range_mhz = end_frequency_mhz - start_frequency_mhz
    num_channels = int(total_range_mhz // channel_bandwidth_mhz)

    if num_channels < 1:
        raise ValueError(
            f"Frequency range ({total_range_mhz} MHz) is smaller than one channel "
            f"({channel_bandwidth_mhz} MHz); no channels can be generated."
        )

    channels = []
    for i in range(num_channels):
        ch_start = start_frequency_mhz + i * channel_bandwidth_mhz
        ch_end = ch_start + channel_bandwidth_mhz
        channels.append({
            "channel_id": i + 1,
            "start_mhz": round(ch_start, 3),
            "end_mhz": round(ch_end, 3),
            "center_mhz": round((ch_start + ch_end) / 2, 3),
            "bandwidth_mhz": round(channel_bandwidth_mhz, 3),
        })

    return channels
