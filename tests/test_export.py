import csv
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.export import write_csv, write_geojson
from pipeline.geo import build_geo_info
from pipeline.schema import Detection, GeoInfo


def make_dets():
    return [
        Detection(id=0, xmin=10, ymin=10, xmax=50, ymax=50, score=0.9, tile_id=0),
        Detection(id=1, xmin=100, ymin=100, xmax=140, ymax=140, score=0.6, tile_id=0),
    ]


def test_write_csv_basic(tmp_path):
    out = str(tmp_path / "out.csv")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_csv(out, make_dets(), geo)

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert float(rows[0]["confidence"]) == 0.9
    assert "crown_area_m2" not in rows[0]  # not available without georeferencing
    assert rows[0]["data_source"] == "live_inference"  # default, honest default


def test_write_csv_labels_precomputed_fallback(tmp_path):
    out = str(tmp_path / "out.csv")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_csv(out, make_dets(), geo, data_source="precomputed_fallback")
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert all(r["data_source"] == "precomputed_fallback" for r in rows)


def test_write_csv_accepts_the_exact_stamp_the_app_uses(tmp_path):
    """app/app.py passes these exact literal strings - pin them here so a
    future refactor can't silently drift away from the required wording."""
    out = str(tmp_path / "out.csv")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_csv(out, make_dets(), geo, data_source="PRECOMPUTED — NOT LIVE")
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["data_source"] == "PRECOMPUTED — NOT LIVE" for r in rows)


def test_write_csv_with_areas(tmp_path):
    out = str(tmp_path / "out.csv")
    geo = GeoInfo(has_crs=True, area_available=True, calibration_source="geotiff_crs",
                  crs_is_geographic=False, source_crs="EPSG:32633", working_crs="EPSG:32633",
                  pixel_size_x_m=0.1, pixel_size_y_m=0.1)
    write_csv(out, make_dets(), geo, crown_areas_m2=[16.0, 16.0])
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["crown_area_m2"] == "16.0"


def test_write_csv_empty_detections_still_writes_header(tmp_path):
    out = str(tmp_path / "out.csv")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_csv(out, [], geo)
    with open(out) as f:
        content = f.read()
    assert "id" in content.splitlines()[0]


def test_write_geojson_no_crs_labels_pixel_coords(tmp_path):
    out = str(tmp_path / "out.geojson")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_geojson(out, make_dets(), geo, raster_transform=None)

    with open(out) as f:
        fc = json.load(f)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    assert "crs_note" in fc  # honest disclosure that these aren't real-world coords
    assert "crs_note" in fc["features"][0]["properties"]
    assert fc["data_source"] == "live_inference"
    assert fc["features"][0]["properties"]["data_source"] == "live_inference"


def test_write_geojson_labels_precomputed_fallback(tmp_path):
    out = str(tmp_path / "out.geojson")
    geo = GeoInfo(has_crs=False, area_available=False)
    write_geojson(out, make_dets(), geo, raster_transform=None, data_source="precomputed_fallback")
    with open(out) as f:
        fc = json.load(f)
    assert fc["data_source"] == "precomputed_fallback"
    assert all(feat["properties"]["data_source"] == "precomputed_fallback" for feat in fc["features"])


def test_write_geojson_with_real_crs_produces_valid_lonlat(tmp_path):
    path = str(tmp_path / "geo.tif")
    width, height, px = 500, 500, 0.1
    transform = from_origin(west=500000, north=5000000, xsize=px, ysize=px)
    data = np.zeros((1, height, width), dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                        dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
        dst.write(data)

    geo = build_geo_info(path)
    out = str(tmp_path / "out2.geojson")
    write_geojson(out, make_dets(), geo, raster_transform=transform)

    with open(out) as f:
        fc = json.load(f)
    assert "crs_note" not in fc
    coords = fc["features"][0]["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    # UTM 33N zone around lon ~15E; sanity-check we're in plausible lon/lat range,
    # not still in raw UTM metres (which would be in the hundreds of thousands).
    assert all(-180 <= lo <= 180 for lo in lons)
    assert all(-90 <= la <= 90 for la in lats)
    assert 10 < lons[0] < 20  # UTM zone 33 spans roughly 12E-18E
