"""Georeferencing utilities.

Central rule this module exists to enforce: NEVER treat a raw pixel size in
degrees as if it were meters, and NEVER fabricate an area number when no
real-world scale is known.

- Projected CRS (units already linear, e.g. UTM metres) -> use the raster
  transform's pixel size directly (converting to metres if the CRS's linear
  unit isn't metres).
- Geographic CRS (lat/lon degrees) -> reproject to an appropriate UTM zone
  and measure actual ground distance per pixel at the image's location.
- No CRS at all -> area is UNAVAILABLE unless the caller supplies a manual
  metres-per-pixel value (from a known camera/drone altitude, or a
  KML-based calibration). No CRS + no manual value => explicitly
  "uncalibrated", never a guessed number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import warnings

import rasterio
from rasterio.errors import NotGeoreferencedWarning
from pyproj import CRS, Transformer

from pipeline.schema import GeoInfo

# pyproj/rasterio linear unit names we might encounter on a projected CRS,
# mapped to their length in metres. Extend as needed; unrecognised units
# fall back to 1.0 with an explicit warning rather than silently guessing.
_LINEAR_UNIT_TO_METRES = {
    "metre": 1.0,
    "meter": 1.0,
    "us survey foot": 0.304800609601219,
    "foot": 0.3048,
    "foot_survey_us": 0.304800609601219,
}


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


@dataclass
class RasterGeoRaw:
    has_crs: bool
    crs: Optional[CRS]
    transform: Optional["rasterio.Affine"]
    width: int
    height: int


def read_raster_geo_raw(path: str) -> RasterGeoRaw:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            crs = src.crs
            has_crs = crs is not None
            return RasterGeoRaw(
                has_crs=has_crs,
                crs=CRS.from_user_input(crs) if has_crs else None,
                transform=src.transform,
                width=src.width,
                height=src.height,
            )


def build_geo_info(
    path: Optional[str],
    manual_mpp: Optional[float] = None,
    has_crs_override: Optional[bool] = None,
) -> GeoInfo:
    """Build a GeoInfo for an input image.

    `path` may be None for non-raster inputs (plain PNG/JPG with no
    embedded georeferencing at all) — in that case only `manual_mpp`
    (metres-per-pixel supplied by the user, e.g. from a known KML distance
    or flight altitude) can make area available.
    """
    notes: List[str] = []

    raw = None
    if path is not None:
        try:
            raw = read_raster_geo_raw(path)
        except Exception as e:  # noqa: BLE001 - surface as a warning, not a crash
            notes.append(f"Could not read raster georeferencing metadata: {e}")
            raw = None

    if raw is not None and raw.has_crs:
        crs = raw.crs
        if crs.is_geographic:
            # Reproject to the appropriate UTM zone to measure REAL ground
            # distance per pixel at the image's own location.
            center_col = raw.width / 2.0
            center_row = raw.height / 2.0
            lon_c, lat_c = raw.transform @ (center_col, center_row)
            working_crs_str = utm_epsg_for_lonlat(lon_c, lat_c)
            transformer = Transformer.from_crs(crs, CRS.from_user_input(working_crs_str), always_xy=True)

            origin_lon, origin_lat = raw.transform @ (center_col, center_row)
            right_lon, right_lat = raw.transform @ (center_col + 1, center_row)
            down_lon, down_lat = raw.transform @ (center_col, center_row + 1)

            ox, oy = transformer.transform(origin_lon, origin_lat)
            rx, ry = transformer.transform(right_lon, right_lat)
            dx, dy = transformer.transform(down_lon, down_lat)

            px_w_m = math.hypot(rx - ox, ry - oy)
            px_h_m = math.hypot(dx - ox, dy - oy)

            notes.append(
                f"Geographic CRS ({crs.to_string()}) reprojected to {working_crs_str} "
                f"to compute real ground distance per pixel (measured at image centre, "
                f"lat={lat_c:.4f}) instead of treating degrees as metres."
            )
            return GeoInfo(
                has_crs=True,
                crs_is_geographic=True,
                source_crs=crs.to_string(),
                working_crs=working_crs_str,
                pixel_size_x_m=px_w_m,
                pixel_size_y_m=px_h_m,
                calibration_source="geotiff_crs",
                area_available=True,
                notes=notes,
            )
        else:
            # Already projected. Confirm the linear unit and convert if needed.
            try:
                unit_name = crs.axis_info[0].unit_name.lower()
            except Exception:  # noqa: BLE001
                unit_name = "metre"
            unit_to_m = _LINEAR_UNIT_TO_METRES.get(unit_name)
            if unit_to_m is None:
                notes.append(
                    f"Projected CRS reports unrecognised linear unit '{unit_name}'; "
                    "assuming metres. Verify this for high-stakes use."
                )
                unit_to_m = 1.0
            px_w = abs(raw.transform.a) * unit_to_m
            px_h = abs(raw.transform.e) * unit_to_m
            notes.append(
                f"Projected CRS ({crs.to_string()}), unit={unit_name}: "
                f"using raster transform pixel size directly."
            )
            return GeoInfo(
                has_crs=True,
                crs_is_geographic=False,
                source_crs=crs.to_string(),
                working_crs=crs.to_string(),
                pixel_size_x_m=px_w,
                pixel_size_y_m=px_h,
                calibration_source="geotiff_crs",
                area_available=True,
                notes=notes,
            )

    # No usable CRS from the raster.
    if manual_mpp is not None and manual_mpp > 0:
        notes.append(
            f"No CRS found in the image; using a user-supplied scale of "
            f"{manual_mpp} m/pixel. Real-world coordinates (GeoJSON) are not "
            f"available in this mode, only areas/counts."
        )
        return GeoInfo(
            has_crs=False,
            crs_is_geographic=None,
            source_crs=None,
            working_crs=None,
            pixel_size_x_m=manual_mpp,
            pixel_size_y_m=manual_mpp,
            calibration_source="manual_mpp",
            area_available=True,
            notes=notes,
        )

    notes.append(
        "No georeference found and no manual scale (metres/pixel) provided. "
        "Area and canopy cover are UNAVAILABLE — reporting a m\u00b2/hectare "
        "figure here would be fabricated. Crown count and pixel-based "
        "positions are still shown."
    )
    return GeoInfo(
        has_crs=False,
        crs_is_geographic=None,
        source_crs=None,
        working_crs=None,
        pixel_size_x_m=None,
        pixel_size_y_m=None,
        calibration_source="none",
        area_available=False,
        notes=notes,
    )


def pixel_polygon_to_working_crs(
    polygon_px: List[Tuple[float, float]],
    geo: GeoInfo,
    raster_transform: Optional["rasterio.Affine"],
    manual_origin_px: Tuple[float, float] = (0.0, 0.0),
) -> List[Tuple[float, float]]:
    """Convert a pixel-space polygon to (x, y) metres in the working CRS.

    - If we have a real CRS + transform: pixel -> source CRS -> working CRS.
    - If we only have a manual metres-per-pixel scale (no CRS at all):
      treat (0, 0) as the local origin and scale directly; this yields a
      LOCAL planar coordinate system (not real-world), which is fine for
      area/cover math but is explicitly not used for GeoJSON real-world
      export (see export.py).
    """
    if geo.has_crs and raster_transform is not None:
        crs_geo = CRS.from_user_input(geo.source_crs)
        pts_src = [raster_transform @ (x, y) for (x, y) in polygon_px]
        if geo.crs_is_geographic:
            transformer = Transformer.from_crs(crs_geo, CRS.from_user_input(geo.working_crs), always_xy=True)
            return [transformer.transform(x, y) for (x, y) in pts_src]
        return pts_src  # already projected & already in metres (post unit conversion upstream)

    if geo.calibration_source == "manual_mpp" and geo.pixel_size_x_m and geo.pixel_size_y_m:
        ox, oy = manual_origin_px
        return [
            ((x - ox) * geo.pixel_size_x_m, (y - oy) * geo.pixel_size_y_m)
            for (x, y) in polygon_px
        ]

    raise ValueError("No calibration available to convert pixel polygon to real-world coordinates")
