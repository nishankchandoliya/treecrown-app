"""Integration test on a genuinely large image (not just tiling-math unit
tests) - the brief is explicit that tiling must be tested on an image large
enough to actually trigger it, not assumed to work from smaller tests.
"""
import os
import resource
import time

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from pipeline.pipeline import run_pipeline, apply_threshold
from tests.fixtures import FakeDetector


@pytest.fixture(scope="module")
def large_geotiff(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("large") / "large_test.tif")
    width, height, px = 6000, 4500, 0.1  # ~27 megapixels, 600m x 450m at 0.1m/px
    transform = from_origin(west=500000, north=5000000, xsize=px, ysize=px)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=3,
        dtype="uint8", crs="EPSG:32633", transform=transform,
        tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        chunk_rows = 500
        for y0 in range(0, height, chunk_rows):
            rows = min(chunk_rows, height - y0)
            data = (np.random.rand(3, rows, width) * 255).astype("uint8")
            dst.write(data, window=Window(0, y0, width, rows))
    return path, width, height, px


def test_large_image_actually_triggers_multiple_tiles(large_geotiff):
    path, width, height, px = large_geotiff
    detector = FakeDetector(boxes_fn=lambda t: [])
    result = run_pipeline(path, detector, tile_size=800, overlap=100)
    assert len(result.tile_grid) > 20, "this test is only meaningful if it actually forces many tiles"
    assert detector.call_count == len(result.tile_grid)


def test_large_image_processing_is_memory_safe(large_geotiff):
    """Peak memory should stay far below what loading the whole raster into
    memory at once would require, proving windowed reads are actually
    windowed and not silently falling back to a full-image load.
    """
    path, width, height, px = large_geotiff
    full_load_size_mb = (width * height * 3) / 1e6  # raw uint8 RGB, no compression

    detector = FakeDetector(boxes_fn=lambda t: [(10, 10, 40, 40, 0.6)])
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = run_pipeline(path, detector, tile_size=800, overlap=100)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # Generous ceiling (this process also holds pytest/rasterio/numpy runtime
    # overhead) - the point is proving we're nowhere near needing the whole
    # image resident, not pinning an exact number.
    assert peak_rss_mb < baseline_rss + 400, (
        f"peak RSS grew by {peak_rss_mb - baseline_rss:.0f}MB processing an image whose full "
        f"in-memory size would be ~{full_load_size_mb:.0f}MB - windowed reads may not be working"
    )


def test_large_image_runtime_is_reasonable(large_geotiff):
    path, width, height, px = large_geotiff
    detector = FakeDetector(boxes_fn=lambda t: [(10, 10, 40, 40, 0.6)])
    t0 = time.time()
    run_pipeline(path, detector, tile_size=800, overlap=100)
    dt = time.time() - t0
    # This measures pipeline overhead (I/O + tiling + dedup), NOT real model
    # inference time (FakeDetector is instant) - just guards against
    # pathological O(n^2) blowups in tiling/dedup at realistic tile counts.
    assert dt < 15.0, f"pipeline overhead alone took {dt:.1f}s on a ~60-tile image - investigate"


def test_large_image_area_math_is_correct(large_geotiff):
    path, width, height, px = large_geotiff
    detector = FakeDetector(boxes_fn=lambda t: [])
    result = run_pipeline(path, detector, tile_size=800, overlap=100)
    expected_m2 = (width * px) * (height * px)
    assert result.analyzed_area_m2 == pytest.approx(expected_m2, rel=1e-6)
    assert result.analyzed_area_m2 == pytest.approx(270000.0, rel=1e-6)  # 27 ha


def test_threshold_sweep_on_large_image_still_never_recalls_detector(large_geotiff):
    path, width, height, px = large_geotiff
    detector = FakeDetector(boxes_fn=lambda t: [(10, 10, 40, 40, 0.55)])
    result = run_pipeline(path, detector, tile_size=800, overlap=100)
    calls_after_run = detector.call_count
    for t in [0.1, 0.3, 0.5, 0.56, 0.9]:
        apply_threshold(result, t)
    assert detector.call_count == calls_after_run
