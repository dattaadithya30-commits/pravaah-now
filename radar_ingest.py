"""
Doppler radar ingestion — reflectivity + radial velocity extraction.

Uses Py-ART (the field-standard Python toolkit for weather radar,
https://arm-doe.github.io/pyart/) to read raw volume-scan files and
grid them onto a regular lat/lon grid usable by the fusion pipeline.

NOTE: exact field names / file format depend on which IMD DWR product
you're handed (ODIM_H5 vs. raw NEXRAD-style vs. vendor format) — the
`refl_field` / `vel_field` names below are Py-ART's common defaults;
adjust to match your actual radar's metadata (`radar.fields.keys()`
will show you what's available once you load a real file).

Install: pip install arm_pyart
"""

from __future__ import annotations

import numpy as np

try:
    import pyart
except ImportError:  # keep the module importable even before pyart is installed
    pyart = None


def load_radar_volume(filepath: str):
    """
    Load a raw radar volume scan file (ODIM_H5, NEXRAD Level II, etc.)
    into a Py-ART Radar object.
    """
    if pyart is None:
        raise ImportError("Install Py-ART first: pip install arm_pyart")
    return pyart.io.read(filepath)


def grid_reflectivity(
    radar,
    grid_shape: tuple[int, int, int] = (1, 256, 256),
    grid_limits: tuple = ((0, 15000), (-128000, 128000), (-128000, 128000)),
    refl_field: str = "reflectivity",
):
    """
    Interpolate a radar volume's reflectivity field onto a regular
    Cartesian grid (the format the fusion pipeline / model expects).

    Args:
        radar: a pyart.core.Radar object (from load_radar_volume).
        grid_shape: (z, y, x) number of grid points.
        grid_limits: ((z_min, z_max), (y_min, y_max), (x_min, x_max)) in meters.
        refl_field: name of the reflectivity field in the radar object.

    Returns:
        np.ndarray of shape (y, x) — a single-level reflectivity grid
        (dBZ), taking the lowest vertical level for a 2D nowcasting input.
    """
    if pyart is None:
        raise ImportError("Install Py-ART first: pip install arm_pyart")

    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        fields=[refl_field],
    )
    # take the lowest vertical level -> (y, x) 2D reflectivity field
    refl_2d = grid.fields[refl_field]["data"][0]
    return np.ma.filled(refl_2d, fill_value=0.0)


def quality_control(refl_grid: np.ndarray, min_valid_dbz: float = -10.0) -> np.ndarray:
    """
    Minimal QC pass: clip noise floor / non-meteorological low values.
    Extend this with real ground-clutter and speckle filters before
    production use — this is a placeholder, not a full QC pipeline.
    """
    cleaned = np.where(refl_grid < min_valid_dbz, 0.0, refl_grid)
    return cleaned


if __name__ == "__main__":
    # Example usage once you have a real radar file to point at:
    #
    #   radar = load_radar_volume("sample_data/IMD_DWR_sample.h5")
    #   refl = grid_reflectivity(radar)
    #   refl = quality_control(refl)
    #   print(refl.shape, refl.min(), refl.max())
    print("radar_ingest.py — see docstring for usage with a real DWR file.")
