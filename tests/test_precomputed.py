import os

from pipeline.precomputed import (
    save_precomputed, load_precomputed, find_precomputed_fallback, precomputed_path_for_image,
)
from pipeline.schema import Detection, GeoInfo, PipelineResult


def make_sample_result():
    geo = GeoInfo(
        has_crs=True, crs_is_geographic=False, source_crs="EPSG:32633", working_crs="EPSG:32633",
        pixel_size_x_m=0.1, pixel_size_y_m=0.1, calibration_source="geotiff_crs",
        area_available=True, notes=["Projected CRS, using transform directly."],
    )
    dets = [
        Detection(id=0, xmin=1, ymin=2, xmax=10, ymax=12, score=0.9, tile_id=0, near_tile_edge=False),
        Detection(id=1, xmin=20, ymin=20, xmax=30, ymax=30, score=0.4, tile_id=1, near_tile_edge=True),
    ]
    return PipelineResult(
        image_path="/some/image.tif", image_width_px=400, image_height_px=400,
        raw_detections=dets, geo=geo, tile_grid=[(0, 0, 400, 400)],
        detector_name="DeepForest (pretrained RetinaNet)", warnings=["1 pre-existing warning"],
        analyzed_area_px2=160000.0, analyzed_area_m2=1600.0,
    )


def test_round_trip_preserves_detections_and_geo(tmp_path):
    result = make_sample_result()
    path = str(tmp_path / "example.precomputed.json")
    save_precomputed(result, path)
    loaded = load_precomputed(path)

    assert loaded.image_width_px == 400
    assert loaded.image_height_px == 400
    assert len(loaded.raw_detections) == 2
    assert loaded.raw_detections[0].score == 0.9
    assert loaded.raw_detections[1].near_tile_edge is True
    assert loaded.geo.has_crs is True
    assert loaded.geo.pixel_size_x_m == 0.1
    assert loaded.analyzed_area_m2 == 1600.0


def test_loaded_result_is_flagged_as_precomputed_not_live(tmp_path):
    result = make_sample_result()
    assert result.used_precomputed_fallback is False  # a fresh live result is NOT flagged

    path = str(tmp_path / "example.precomputed.json")
    save_precomputed(result, path)
    loaded = load_precomputed(path)

    assert loaded.used_precomputed_fallback is True
    assert any("precomputed" in w.lower() for w in loaded.warnings)


def test_find_precomputed_fallback_missing_returns_none(tmp_path):
    found = find_precomputed_fallback("/some/image.tif", str(tmp_path))
    assert found is None


def test_find_precomputed_fallback_present(tmp_path):
    result = make_sample_result()
    expected_path = precomputed_path_for_image("/anywhere/OSBS_029.tif", str(tmp_path))
    save_precomputed(result, expected_path)

    found = find_precomputed_fallback("/different/dir/OSBS_029.tif", str(tmp_path))
    assert found == expected_path


def test_precomputed_path_uses_basename_only():
    p1 = precomputed_path_for_image("/a/b/OSBS_029.tif", "/precomp")
    p2 = precomputed_path_for_image("/x/y/z/OSBS_029.tif", "/precomp")
    assert p1 == p2  # matched by filename, not full path (image may be re-uploaded to a temp dir)
