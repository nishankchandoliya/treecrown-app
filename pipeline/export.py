"""CSV and GeoJSON export for a thresholded detection set."""
from __future__ import annotations

import csv
import json
from typing import List, Optional

from pyproj import CRS, Transformer

from pipeline.geo import pixel_polygon_to_working_crs
from pipeline.schema import Detection, GeoInfo


def detections_to_csv_rows(
    detections: List[Detection], geo: GeoInfo, crown_areas_m2: Optional[List[float]] = None,
    data_source: str = "live_inference",
):
    rows = []
    for i, d in enumerate(detections):
        cx, cy = d.centroid_px
        row = {
            "id": d.id,
            "xmin_px": round(d.xmin, 2),
            "ymin_px": round(d.ymin, 2),
            "xmax_px": round(d.xmax, 2),
            "ymax_px": round(d.ymax, 2),
            "centroid_x_px": round(cx, 2),
            "centroid_y_px": round(cy, 2),
            "confidence": round(d.score, 4),
            "tile_id": d.tile_id,
            "near_tile_edge": d.near_tile_edge,
            "data_source": data_source,
        }
        if geo.area_available and crown_areas_m2 is not None:
            row["crown_area_m2"] = round(crown_areas_m2[i], 3)
        rows.append(row)
    return rows


def write_csv(
    path: str, detections: List[Detection], geo: GeoInfo, crown_areas_m2: Optional[List[float]] = None,
    data_source: str = "live_inference",
):
    rows = detections_to_csv_rows(detections, geo, crown_areas_m2, data_source=data_source)
    if not rows:
        fieldnames = ["id", "xmin_px", "ymin_px", "xmax_px", "ymax_px", "centroid_x_px",
                      "centroid_y_px", "confidence", "tile_id", "near_tile_edge", "data_source"]
    else:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(
    path: str,
    detections: List[Detection],
    geo: GeoInfo,
    raster_transform,
    crown_areas_m2: Optional[List[float]] = None,
    data_source: str = "live_inference",
):
    """Writes real-world GeoJSON (EPSG:4326 lon/lat) when the image has a
    CRS. If there's no CRS at all, we cannot honestly claim real-world
    coordinates — in that case this writes a LOCAL pixel-based GeoJSON
    (still valid GeoJSON, but with a `"crs_note"` in each feature's
    properties making clear these are pixel coordinates, not geographic
    ones) rather than pretending we know where on Earth this is.
    """
    features = []

    to_wgs84 = None
    if geo.has_crs:
        working_crs = CRS.from_user_input(geo.working_crs)
        if working_crs.to_epsg() != 4326:
            to_wgs84 = Transformer.from_crs(working_crs, CRS.from_epsg(4326), always_xy=True)

    for i, d in enumerate(detections):
        poly_px = d.effective_polygon_px()
        if geo.has_crs:
            coords_m = pixel_polygon_to_working_crs(poly_px, geo, raster_transform)
            if to_wgs84 is not None:
                coords_out = [to_wgs84.transform(x, y) for (x, y) in coords_m]
            else:
                coords_out = coords_m
            crs_note = None
        else:
            coords_out = poly_px  # raw pixel coords, NOT geographic
            crs_note = "No georeference available: coordinates are image pixel positions, not real-world lon/lat."

        props = {
            "id": d.id,
            "confidence": round(d.score, 4),
            "tile_id": d.tile_id,
            "data_source": data_source,
        }
        if crown_areas_m2 is not None and geo.area_available:
            props["crown_area_m2"] = round(crown_areas_m2[i], 3)
        if crs_note:
            props["crs_note"] = crs_note

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[x, y] for (x, y) in coords_out]]},
                "properties": props,
            }
        )

    fc = {"type": "FeatureCollection", "features": features, "data_source": data_source}
    if not geo.has_crs:
        fc["crs_note"] = "No georeference in source image — geometries are in PIXEL coordinates, not lon/lat."

    with open(path, "w") as f:
        json.dump(fc, f)
