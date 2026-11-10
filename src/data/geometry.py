"""WGS84 coordinates and CATS-CF sensor geometry."""

import numpy as np
import math

BASE_HZ = 125
GAP_OFFSET_M = 3.0
WGS84_A_M = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(
    lat_deg: np.ndarray, lon_deg: np.ndarray, height_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    height = np.asarray(height_m, dtype=float)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    prime_vertical_radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (prime_vertical_radius + height) * cos_lat * np.cos(lon)
    y = (prime_vertical_radius + height) * cos_lat * np.sin(lon)
    z = (prime_vertical_radius * (1.0 - WGS84_E2) + height) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    x0: float,
    y0: float,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    dx, dy, dz = x - x0, y - y0, z - z0
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up
