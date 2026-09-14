"""Shared data structures for the tree-crown detection pipeline.

Kept dependency-free (stdlib + dataclasses only) so every other module
can import this without pulling in rasterio/shapely/torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """A single detected tree crown, in GLOBAL image pixel coordinates."""

    id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    tile_id: int
    near_tile_edge: bool = False
    # Polygon in pixel coords, (x, y) tuples. Defaults to the box corners
    # when no finer crown-shape refinement is available.
    polygon_px: Optional[List[Tuple[float, float]]] = None

    def box_polygon_px(self) -> List[Tuple[float, float]]:
        return [
            (self.xmin, self.ymin),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
            (self.xmin, self.ymax),
            (self.xmin, self.ymin),
        ]

    def effective_polygon_px(self) -> List[Tuple[float, float]]:
        return self.polygon_px if self.polygon_px is not None else self.box_polygon_px()

    @property
    def centroid_px(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width_px(self) -> float:
        return self.xmax - self.xmin

    @property
    def height_px(self) -> float:
        return self.ymax - self.ymin


@dataclass
class GeoInfo:
    """Georeferencing status/derivation for one input image."""

    has_crs: bool
    crs_is_geographic: Optional[bool] = None  # None if has_crs is False
    source_crs: Optional[str] = None  # e.g. "EPSG:4326"
    working_crs: Optional[str] = None  # projected CRS actually used for area math
    pixel_size_x_m: Optional[float] = None
    pixel_size_y_m: Optional[float] = None
    calibration_source: str = "none"  # "geotiff_crs" | "manual_mpp" | "none"
    area_available: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Everything downstream (UI, exports) needs, computed ONCE per image.

    `raw_detections` holds every candidate at every confidence score the
    model produced. Nothing here depends on a chosen confidence threshold —
    that filtering happens later, cheaply, in `apply_threshold`.
    """

    image_path: str
    image_width_px: int
    image_height_px: int
    raw_detections: List[Detection]
    geo: GeoInfo
    tile_grid: List[Tuple[int, int, int, int]]  # (x_off, y_off, w, h)
    detector_name: str
    warnings: List[str] = field(default_factory=list)
    analyzed_area_px2: Optional[float] = None  # width*height of analyzed region
    analyzed_area_m2: Optional[float] = None
    used_precomputed_fallback: bool = False


@dataclass
class ThresholdedView:
    """Result of applying a confidence threshold to an existing PipelineResult.

    Computing this must NEVER re-run detection.
    """

    threshold: float
    detections: List[Detection]
    count: int
    union_area_m2: Optional[float]
    union_area_px2: float
    canopy_cover_pct: Optional[float]
    score_histogram: List[float]  # raw scores of ALL raw detections (for context)
