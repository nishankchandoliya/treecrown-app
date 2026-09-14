import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from pipeline.pipeline import run_pipeline, apply_threshold, compute_quality
from pipeline.image_io import ImageOpenError
from tests.fixtures import FakeDetector


def write_plain_png(path, width=1200, height=900):
    arr = (np.random.rand(height, width, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def write_geotiff_utm(path, width=1200, height=900, px_size=0.1):
    data = (np.random.rand(1, height, width) * 255).astype("uint8")
    transform = from_origin(west=500000, north=5000000, xsize=px_size, ysize=px_size)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs="EPSG:32633", transform=transform,
    ) as dst:
        dst.write(data)


def test_detector_called_exactly_once_per_tile_and_never_again_on_threshold_change(tmp_path):
    path = str(tmp_path / "img.png")
    write_plain_png(path, width=1200, height=900)  # triggers multiple 800px tiles

    # every tile "detects" one fixed box at a fixed local position + varying confidence
    def boxes_fn(tile_img):
        return [(10.0, 10.0, 60.0, 60.0, 0.75)]

    detector = FakeDetector(boxes_fn=boxes_fn)
    result = run_pipeline(path, detector, tile_size=800, overlap=100)

    calls_after_run = detector.call_count
    assert calls_after_run == len(result.tile_grid)
    assert calls_after_run > 1  # confirm this test actually exercised multi-tile tiling

    # Sweep the threshold repeatedly - detector must NOT be called again.
    for t in [0.0, 0.3, 0.5, 0.75, 0.76, 0.9, 1.0]:
        apply_threshold(result, t)

    assert detector.call_count == calls_after_run, "confidence threshold changes must never re-run detection"


def test_threshold_filters_monotonically(tmp_path):
    path = str(tmp_path / "img.png")
    write_plain_png(path, width=700, height=700)

    def boxes_fn(tile_img):
        return [
            (10, 10, 50, 50, 0.2),
            (100, 100, 140, 140, 0.5),
            (200, 200, 240, 240, 0.9),
        ]

    detector = FakeDetector(boxes_fn=boxes_fn)
    result = run_pipeline(path, detector, tile_size=800, overlap=100)

    v_low = apply_threshold(result, 0.0)
    v_mid = apply_threshold(result, 0.4)
    v_high = apply_threshold(result, 0.85)

    assert v_low.count == 3
    assert v_mid.count == 2
    assert v_high.count == 1
    # raising the threshold can only remove detections, never add
    assert v_low.count >= v_mid.count >= v_high.count


def test_georeferenced_image_produces_real_area(tmp_path):
    path = str(tmp_path / "geo.tif")
    write_geotiff_utm(path, width=1000, height=1000, px_size=0.1)  # 0.1 m/px -> 100m x 100m image = 10,000 m2

    def boxes_fn(tile_img):
        return [(0, 0, 100, 100, 0.9)]  # a 100x100 PIXEL box -> 10x10 m -> 100 m^2 crown

    detector = FakeDetector(boxes_fn=boxes_fn)
    result = run_pipeline(path, detector, tile_size=800, overlap=100)

    assert result.geo.area_available is True
    assert result.analyzed_area_m2 == pytest.approx(1000 * 0.1 * 1000 * 0.1, rel=1e-6)  # 10,000 m2

    view = apply_threshold(result, 0.0, raster_transform=result.geo and _get_transform(path))
    assert view.union_area_m2 is not None
    assert view.canopy_cover_pct is not None
    assert view.canopy_cover_pct > 0


def _get_transform(path):
    with rasterio.open(path) as src:
        return src.transform


def test_no_crs_image_still_gives_cover_pct_but_no_absolute_area(tmp_path):
    path = str(tmp_path / "plain.png")
    write_plain_png(path, width=700, height=700)

    def boxes_fn(tile_img):
        return [(0, 0, 100, 100, 0.9)]

    detector = FakeDetector(boxes_fn=boxes_fn)
    result = run_pipeline(path, detector, tile_size=800, overlap=100)

    assert result.geo.area_available is False
    assert result.analyzed_area_m2 is None

    view = apply_threshold(result, 0.0)
    assert view.canopy_cover_pct is not None  # ratio is always computable
    assert view.union_area_m2 is None  # absolute area correctly withheld, not fabricated


def test_no_detections_does_not_crash(tmp_path):
    path = str(tmp_path / "empty.png")
    write_plain_png(path, width=500, height=500)

    detector = FakeDetector(boxes_fn=lambda tile: [])
    result = run_pipeline(path, detector, tile_size=800, overlap=100)

    assert result.raw_detections == []
    assert any("no trees" in w.lower() for w in result.warnings)

    view = apply_threshold(result, 0.5)
    assert view.count == 0
    assert view.canopy_cover_pct == 0.0 or view.canopy_cover_pct is None


def test_corrupted_file_raises_clear_error_not_a_crash(tmp_path):
    path = str(tmp_path / "corrupt.tif")
    with open(path, "wb") as f:
        f.write(b"this is not a real image file, just garbage bytes")

    detector = FakeDetector(fixed_boxes=[])
    with pytest.raises(ImageOpenError):
        run_pipeline(path, detector)


def test_quality_indicator_runs_end_to_end(tmp_path):
    path = str(tmp_path / "geo2.tif")
    write_geotiff_utm(path, width=1000, height=1000, px_size=0.1)

    detector = FakeDetector(boxes_fn=lambda t: [(0, 0, 50, 50, 0.85)])
    result = run_pipeline(path, detector, tile_size=800, overlap=100)
    view = apply_threshold(result, 0.5, raster_transform=_get_transform(path))
    q = compute_quality(result, view)
    assert q.label in {"Good", "Moderate", "Limited"}
    assert len(q.reasons) > 0
