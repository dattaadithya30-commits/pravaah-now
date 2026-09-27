"""
Copernicus DEM feature extraction — slope & aspect for orographic-lift
modeling (referenced on Slide 2/3: "Uses DEM slopes to map
mountain-lift zones").

Slope/aspect are computed via the standard Horn (1981) method used in
GIS tools like ArcGIS/QGIS — implemented directly with NumPy gradients
here so it's dependency-light (only needs rasterio to read the file).

Install: pip install rasterio numpy
"""

from __future__ import annotations

import numpy as np
import rasterio


def load_dem(filepath: str) -> tuple[np.ndarray, float, float]:
    """
    Load a Copernicus DEM GeoTIFF.

    Returns:
        elevation: (H, W) array in meters.
        pixel_size_x, pixel_size_y: ground resolution in meters
            (needed to compute a correctly-scaled slope).
    """
    with rasterio.open(filepath) as src:
        elevation = src.read(1).astype(np.float32)
        pixel_size_x = abs(src.transform.a)
        pixel_size_y = abs(src.transform.e)
    return elevation, pixel_size_x, pixel_size_y


def compute_slope_aspect(
    elevation: np.ndarray, pixel_size_x: float, pixel_size_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees, 0=North, clockwise)
    from a DEM elevation array.

    Slope feeds the orographic-lift term in the fusion grid: steeper
    terrain facing an incoming moist flow enhances rainfall
    (orographic enhancement) — exactly the effect pure radar/satellite
    models miss and that DEM conditioning is meant to correct for.
    """
    dz_dy, dz_dx = np.gradient(elevation, pixel_size_y, pixel_size_x)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 90.0 - aspect_deg + 360.0)
    aspect_deg = np.mod(aspect_deg, 360.0)

    return slope_deg, aspect_deg


def orographic_lift_factor(
    slope_deg: np.ndarray, aspect_deg: np.ndarray, wind_dir_deg: float
) -> np.ndarray:
    """
    A simple, explainable windward-slope enhancement factor: terrain
    facing INTO the wind gets a positive multiplier, leeward terrain
    gets a reduction (rain-shadow effect). This is the same physical
    mechanism used as the fallback "BCSD-style" proxy in our slide
    3/technical-approach discussion, and doubles as a sanity-check
    feature the full ConvLSTM model can learn to refine.

    Args:
        slope_deg, aspect_deg: from compute_slope_aspect().
        wind_dir_deg: prevailing/storm-relative wind direction the
            slope is being evaluated against (0=North, clockwise).

    Returns:
        A per-pixel multiplier, roughly in [0.7, 1.3] — windward steep
        slopes enhanced, leeward slopes reduced, flat terrain ~1.0.
    """
    angle_diff = np.abs(aspect_deg - wind_dir_deg)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # shortest angular distance

    # windward (facing the wind, angle_diff near 0) -> enhancement
    # leeward (angle_diff near 180) -> reduction
    directional_term = np.cos(np.radians(angle_diff))  # +1 windward, -1 leeward
    slope_term = np.clip(slope_deg / 30.0, 0, 1)  # saturate beyond ~30 degrees

    factor = 1.0 + 0.3 * directional_term * slope_term
    return factor


if __name__ == "__main__":
    # Example usage once you have a real Copernicus DEM tile:
    #
    #   elev, px, py = load_dem("sample_data/copernicus_dem_pune.tif")
    #   slope, aspect = compute_slope_aspect(elev, px, py)
    #   lift = orographic_lift_factor(slope, aspect, wind_dir_deg=270)
    print("dem_utils.py — see docstring for usage with a real Copernicus DEM tile.")
