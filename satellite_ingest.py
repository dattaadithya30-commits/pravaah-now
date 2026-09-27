"""
INSAT-3D/3DR ingestion via MOSDAC.

MOSDAC distributes INSAT-3D/3DR L1/L1B products as HDF5 files. Exact
dataset/group names vary by product (e.g. cloud-top temperature vs.
water-vapor channel vs. rainfall estimate) — inspect a real downloaded
file with `h5py` (see `inspect_file` below) and adjust `CHANNEL_KEYS`
to match before this is used against real data.

MOSDAC access: https://mosdac.gov.in (registration required — team
already has access, per our PPT's Data Sources / References slides).

Install: pip install h5py
"""

from __future__ import annotations

import h5py
import numpy as np


# Adjust these once you've inspected a real MOSDAC file's structure —
# these are placeholder guesses based on common INSAT-3D product naming.
CHANNEL_KEYS = {
    "cloud_top_temp": "IMG_TIR1",
    "water_vapor": "IMG_WV",
    "visible": "IMG_VIS",
}


def inspect_file(filepath: str) -> list[str]:
    """
    Print/return every dataset path in an HDF5 file — run this FIRST
    against a real downloaded MOSDAC file to find the correct keys,
    then update CHANNEL_KEYS above.
    """
    paths = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(f"{name}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(filepath, "r") as f:
        f.visititems(visitor)

    for p in paths:
        print(p)
    return paths


def load_channel(filepath: str, channel: str) -> np.ndarray:
    """
    Load a single INSAT channel as a numpy array.

    Args:
        filepath: path to the downloaded MOSDAC HDF5 file.
        channel: one of CHANNEL_KEYS' keys (e.g. "cloud_top_temp").
    """
    if channel not in CHANNEL_KEYS:
        raise ValueError(f"Unknown channel '{channel}'. Options: {list(CHANNEL_KEYS)}")

    key = CHANNEL_KEYS[channel]
    with h5py.File(filepath, "r") as f:
        if key not in f:
            raise KeyError(
                f"Dataset '{key}' not found in file. Run inspect_file() "
                f"first to check the real structure of this product."
            )
        data = f[key][:]
    return data


def normalize_brightness_temp(bt_kelvin: np.ndarray, t_min: float = 180.0, t_max: float = 320.0) -> np.ndarray:
    """
    Normalize brightness temperature to [0, 1] for model input.
    Colder cloud tops (lower T) generally indicate deeper, more
    vigorous convection — this is a standard preprocessing step
    before feeding into the fusion grid.
    """
    clipped = np.clip(bt_kelvin, t_min, t_max)
    return (clipped - t_min) / (t_max - t_min)


if __name__ == "__main__":
    # Example usage once you have a real downloaded MOSDAC file:
    #
    #   inspect_file("sample_data/INSAT3DR_sample.h5")
    #   ctt = load_channel("sample_data/INSAT3DR_sample.h5", "cloud_top_temp")
    #   ctt_norm = normalize_brightness_temp(ctt)
    print("satellite_ingest.py — run inspect_file() against a real MOSDAC file first.")
