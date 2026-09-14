"""Orchestrates: load -> tile -> detect -> cross-tile dedup -> georeference
-> cache. Confidence-threshold changes NEVER re-enter this file's detection
loop — see apply_threshold, which only reads from an already-built
PipelineResult.
"""
from __future__ import annotations

import statistics
from typing import List, Optional

from pipeline import canopy, dedup, geo, quality, tiling
from pipeline.detector import BaseDetector
from pipeline.image_io import ImageOpenError, ImageReader
from pipeline.schema import Detection, GeoInfo, PipelineResult, ThresholdedView

DEFAULT_TILE_SIZE = 800
DEFAULT_OVERLAP = 100
DEFAULT_NMS_IOU = 0.4
SAFETY_MAX_TILES = 400  # soft ceiling: warn, don't silently drop data


def run_pipeline(
    image_path: str,
    detector: BaseDetector,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    nms_iou: float = DEFAULT_NMS_IOU,
    manual_mpp: Optional[float] = None,
) -> PipelineResult:
    warnings: List[str] = []

    try:
        reader = ImageReader(image_path)
    except ImageOpenError as e:
        raise  # let the caller (UI) show this directly - it's already a clear message

    with reader:
        width, height = reader.width, reader.height

        geo_info = geo.build_geo_info(image_path, manual_mpp=manual_mpp)
        warnings.extend(geo_info.notes)

        grid = tiling.compute_tile_grid(width, height, tile_size, overlap)
        if len(grid) > SAFETY_MAX_TILES:
            warnings.append(
                f"Image requires {len(grid)} tiles at the current tile size, above this "
                f"environment's soft safety threshold ({SAFETY_MAX_TILES}). Proceeding, but "
                f"this may be slow on limited hardware. Consider a larger tile size or a "
                f"smaller crop for faster iteration."
            )

        all_local_dets = []
        det_id = 0
        raw_detections: List[Detection] = []

        for tile_id, (x_off, y_off, tw, th) in enumerate(grid):
            tile_img = reader.read_window(x_off, y_off, tw, th)
            local_dets = detector.predict_tile(tile_img)  # [(xmin,ymin,xmax,ymax,score), ...]
            global_dets = tiling.local_to_global(local_dets, x_off, y_off, tile_id, width, height)
            for gd in global_dets:
                near_edge = tiling.flag_near_edge(
                    gd["xmin"], gd["ymin"], gd["xmax"], gd["ymax"], x_off, y_off, tw, th
                )
                raw_detections.append(
                    Detection(
                        id=det_id,
                        xmin=gd["xmin"], ymin=gd["ymin"], xmax=gd["xmax"], ymax=gd["ymax"],
                        score=gd["score"], tile_id=tile_id, near_tile_edge=near_edge,
                    )
                )
                det_id += 1

        # Cross-tile dedup runs ONCE, on the full raw set, right after detection.
        merged = dedup.merge_duplicates(raw_detections, iou_threshold=nms_iou)
        # Re-number ids after merge for a clean, stable export order.
        for new_id, d in enumerate(merged):
            d.id = new_id

        analyzed_area_px2 = float(width) * float(height)
        analyzed_area_m2 = None
        if geo_info.area_available and geo_info.pixel_size_x_m and geo_info.pixel_size_y_m:
            analyzed_area_m2 = analyzed_area_px2 * geo_info.pixel_size_x_m * geo_info.pixel_size_y_m

        if not merged:
            warnings.append("No trees were detected in this image at any confidence score.")

        return PipelineResult(
            image_path=image_path,
            image_width_px=width,
            image_height_px=height,
            raw_detections=merged,
            geo=geo_info,
            tile_grid=grid,
            detector_name=getattr(detector, "name", detector.__class__.__name__),
            warnings=warnings,
            analyzed_area_px2=analyzed_area_px2,
            analyzed_area_m2=analyzed_area_m2,
        )


def apply_threshold(result: PipelineResult, threshold: float, raster_transform=None) -> ThresholdedView:
    """Pure post-processing of an EXISTING PipelineResult. Never touches the
    detector. Safe to call on every slider move.
    """
    filtered = [d for d in result.raw_detections if d.score >= threshold]

    # Canopy cover % is always computable from pixel-space polygons alone
    # (it's a ratio, so absolute scale cancels out) — this is a measured
    # geometric fact about the image, not something that needs calibration.
    px_polys = [d.effective_polygon_px() for d in filtered]
    union_px2, _naive_sum_px2 = canopy.union_area(px_polys)
    cover_pct = canopy.canopy_cover_pct(union_px2, result.analyzed_area_px2)

    union_area_m2 = None
    if result.geo.area_available and raster_transform is not None:
        real_polys = [
            geo.pixel_polygon_to_working_crs(d.effective_polygon_px(), result.geo, raster_transform)
            for d in filtered
        ]
        union_area_m2, _ = canopy.union_area(real_polys)
    elif result.geo.area_available and result.geo.calibration_source == "manual_mpp":
        real_polys = [
            geo.pixel_polygon_to_working_crs(d.effective_polygon_px(), result.geo, None)
            for d in filtered
        ]
        union_area_m2, _ = canopy.union_area(real_polys)

    return ThresholdedView(
        threshold=threshold,
        detections=filtered,
        count=len(filtered),
        union_area_m2=union_area_m2,
        union_area_px2=union_px2,
        canopy_cover_pct=cover_pct,
        score_histogram=[d.score for d in result.raw_detections],
    )


def compute_quality(result: PipelineResult, view: ThresholdedView) -> quality.QualityResult:
    n = len(result.raw_detections)
    if n == 0:
        pct_near_edge = 0.0
        median_conf = None
        pct_low_conf = 0.0
    else:
        pct_near_edge = 100.0 * sum(1 for d in result.raw_detections if d.near_tile_edge) / n
        median_conf = statistics.median(d.score for d in result.raw_detections)
        pct_low_conf = 100.0 * sum(1 for d in result.raw_detections if d.score < 0.3) / n

    overlap_ratio = None
    if view.detections:
        px_polys = [d.effective_polygon_px() for d in view.detections]
        union_px2, naive_sum_px2 = canopy.union_area(px_polys)
        if naive_sum_px2 > 0:
            overlap_ratio = union_px2 / naive_sum_px2

    factors = quality.QualityFactors(
        has_georeference=result.geo.has_crs,
        resolution_m_per_px=result.geo.pixel_size_x_m,
        pct_detections_near_edge=pct_near_edge,
        median_confidence=median_conf,
        pct_low_confidence=pct_low_conf,
        crown_overlap_ratio=overlap_ratio,
    )
    return quality.assess_quality(factors)
