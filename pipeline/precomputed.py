"""Save/load a PipelineResult to/from disk, for the ONE sanctioned use case
in the brief: if live inference cannot reliably run on the deployment
platform at demo time, fall back to a precomputed result - but ONLY with
an explicit, impossible-to-miss "these are precomputed, not live" label.

This module does not decide WHEN to use a fallback - that's a UI-level,
user-visible choice (see app/app.py). It just does the (de)serialization
correctly and marks the result so nothing downstream can present it as
live by accident.
"""
from __future__ import annotations

import json
from typing import Optional

from pipeline.schema import Detection, GeoInfo, PipelineResult


def serialize_pipeline_result(result: PipelineResult) -> dict:
    return {
        "image_path": result.image_path,
        "image_width_px": result.image_width_px,
        "image_height_px": result.image_height_px,
        "detector_name": result.detector_name,
        "warnings": result.warnings,
        "analyzed_area_px2": result.analyzed_area_px2,
        "analyzed_area_m2": result.analyzed_area_m2,
        "tile_grid": result.tile_grid,
        "geo": {
            "has_crs": result.geo.has_crs,
            "crs_is_geographic": result.geo.crs_is_geographic,
            "source_crs": result.geo.source_crs,
            "working_crs": result.geo.working_crs,
            "pixel_size_x_m": result.geo.pixel_size_x_m,
            "pixel_size_y_m": result.geo.pixel_size_y_m,
            "calibration_source": result.geo.calibration_source,
            "area_available": result.geo.area_available,
            "notes": result.geo.notes,
        },
        "detections": [
            {
                "id": d.id, "xmin": d.xmin, "ymin": d.ymin, "xmax": d.xmax, "ymax": d.ymax,
                "score": d.score, "tile_id": d.tile_id, "near_tile_edge": d.near_tile_edge,
                "polygon_px": d.polygon_px,
            }
            for d in result.raw_detections
        ],
    }


def save_precomputed(result: PipelineResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(serialize_pipeline_result(result), f, indent=2)


def load_precomputed(path: str) -> PipelineResult:
    with open(path) as f:
        raw = json.load(f)

    geo_raw = raw["geo"]
    geo = GeoInfo(
        has_crs=geo_raw["has_crs"],
        crs_is_geographic=geo_raw["crs_is_geographic"],
        source_crs=geo_raw["source_crs"],
        working_crs=geo_raw["working_crs"],
        pixel_size_x_m=geo_raw["pixel_size_x_m"],
        pixel_size_y_m=geo_raw["pixel_size_y_m"],
        calibration_source=geo_raw["calibration_source"],
        area_available=geo_raw["area_available"],
        notes=geo_raw["notes"],
    )
    detections = [
        Detection(
            id=d["id"], xmin=d["xmin"], ymin=d["ymin"], xmax=d["xmax"], ymax=d["ymax"],
            score=d["score"], tile_id=d["tile_id"], near_tile_edge=d["near_tile_edge"],
            polygon_px=d.get("polygon_px"),
        )
        for d in raw["detections"]
    ]

    return PipelineResult(
        image_path=raw["image_path"],
        image_width_px=raw["image_width_px"],
        image_height_px=raw["image_height_px"],
        raw_detections=detections,
        geo=geo,
        tile_grid=[tuple(t) for t in raw["tile_grid"]],
        detector_name=raw["detector_name"],
        warnings=raw["warnings"] + ["These are PRECOMPUTED results captured earlier - not a live run."],
        analyzed_area_px2=raw["analyzed_area_px2"],
        analyzed_area_m2=raw["analyzed_area_m2"],
        used_precomputed_fallback=True,  # the one field that matters most here
    )


def precomputed_path_for_image(image_path: str, precomputed_dir: str) -> str:
    import os
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(precomputed_dir, f"{base}.precomputed.json")


def find_precomputed_fallback(image_path: str, precomputed_dir: str) -> Optional[str]:
    import os
    candidate = precomputed_path_for_image(image_path, precomputed_dir)
    return candidate if os.path.exists(candidate) else None
