import math
import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.geo import build_geo_info, utm_epsg_for_lonlat, pixel_polygon_to_working_crs


def write_synthetic_geotiff(path, crs, transform, width=100, height=100):
    data = np.zeros((1, height, width), dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1,
        dtype="uint8", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)


def test_utm_zone_selection_northern_hemisphere():
    # London: lon ~ -0.1, lat ~ 51.5 -> UTM zone 30N -> EPSG:32630
    assert utm_epsg_for_lonlat(-0.1, 51.5) == "EPSG:32630"


def test_utm_zone_selection_southern_hemisphere():
    # Sao Paulo: lon ~ -46.6, lat ~ -23.5 -> zone 23S -> EPSG:32723
    assert utm_epsg_for_lonlat(-46.6, -23.5) == "EPSG:32723"


def test_geographic_crs_does_not_treat_degrees_as_metres(tmp_path):
    """The critical bug this whole module exists to prevent: a raster in
    EPSG:4326 with e.g. 0.0001-degree pixels must NOT report a pixel size
    of "0.0001 metres". It must report the real ground distance, which at
    the equator is roughly 0.0001 * 111320 ~= 11.1 m.
    """
    path = str(tmp_path / "equator.tif")
    # 0.0001 deg/pixel, centred near the equator (small bbox around lon=0, lat=0)
    transform = from_origin(west=-0.005, north=0.005, xsize=0.0001, ysize=0.0001)
    write_synthetic_geotiff(path, crs="EPSG:4326", transform=transform, width=100, height=100)

    info = build_geo_info(path)
    assert info.has_crs is True
    assert info.crs_is_geographic is True
    assert info.area_available is True

    # Must be metres-scale (tens of metres), NOT 0.0001 (which would be the
    # degrees-as-metres bug) and not wildly different from the expected
    # ~11.1 m/pixel at the equator.
    assert info.pixel_size_x_m is not None
    assert 9.0 < info.pixel_size_x_m < 13.0, f"got {info.pixel_size_x_m} - looks like degrees leaked through as metres"
    assert 9.0 < info.pixel_size_y_m < 13.0


def test_geographic_crs_pixel_size_shrinks_at_high_latitude():
    """At high latitude, a fixed degree-of-longitude spans much less ground
    distance than at the equator (shrinks by cos(latitude)). If our code
    just used a constant degrees->metres factor everywhere, this test would
    catch it."""
    import tempfile, os

    d = tempfile.mkdtemp()
    equator_path = os.path.join(d, "equator.tif")
    high_lat_path = os.path.join(d, "high_lat.tif")

    xsize = ysize = 0.001
    t_eq = from_origin(west=-0.05, north=0.05, xsize=xsize, ysize=ysize)
    write_synthetic_geotiff(equator_path, "EPSG:4326", t_eq, width=100, height=100)

    # centred near lat=70N instead of the equator
    t_hi = from_origin(west=-0.05, north=70.05, xsize=xsize, ysize=ysize)
    write_synthetic_geotiff(high_lat_path, "EPSG:4326", t_hi, width=100, height=100)

    info_eq = build_geo_info(equator_path)
    info_hi = build_geo_info(high_lat_path)

    # Longitude-direction pixel size should shrink noticeably at high latitude;
    # cos(70deg) ~= 0.342, so expect roughly a third of the equatorial value.
    ratio = info_hi.pixel_size_x_m / info_eq.pixel_size_x_m
    assert 0.25 < ratio < 0.45, f"expected ~cos(70deg)=0.342 ratio, got {ratio}"

    # Latitude-direction (north-south) pixel size shouldn't shrink nearly as
    # much with latitude (meridians converge much less than parallels here).
    ratio_y = info_hi.pixel_size_y_m / info_eq.pixel_size_y_m
    assert 0.9 < ratio_y < 1.1


def test_projected_crs_uses_transform_directly(tmp_path):
    path = str(tmp_path / "utm.tif")
    # UTM zone 33N, 0.5 m pixels - already in metres, no reprojection needed
    transform = from_origin(west=500000, north=5000000, xsize=0.5, ysize=0.5)
    write_synthetic_geotiff(path, crs="EPSG:32633", transform=transform, width=100, height=100)

    info = build_geo_info(path)
    assert info.has_crs is True
    assert info.crs_is_geographic is False
    assert info.calibration_source == "geotiff_crs"
    assert math.isclose(info.pixel_size_x_m, 0.5, rel_tol=1e-6)
    assert math.isclose(info.pixel_size_y_m, 0.5, rel_tol=1e-6)


def test_no_crs_and_no_manual_scale_is_explicitly_unavailable(tmp_path):
    from PIL import Image
    path = str(tmp_path / "plain.png")
    Image.fromarray(np.zeros((50, 60, 3), dtype="uint8")).save(path)

    info = build_geo_info(path, manual_mpp=None)
    assert info.has_crs is False
    assert info.area_available is False
    assert info.pixel_size_x_m is None
    assert any("UNAVAILABLE" in n or "unavailable" in n.lower() for n in info.notes)


def test_no_crs_but_manual_mpp_makes_area_available(tmp_path):
    from PIL import Image
    path = str(tmp_path / "plain2.png")
    Image.fromarray(np.zeros((50, 60, 3), dtype="uint8")).save(path)

    info = build_geo_info(path, manual_mpp=0.08)
    assert info.has_crs is False
    assert info.area_available is True
    assert info.calibration_source == "manual_mpp"
    assert info.pixel_size_x_m == 0.08


def test_pixel_polygon_to_working_crs_manual_scale():
    from pipeline.schema import GeoInfo
    geo_info = GeoInfo(
        has_crs=False, area_available=True, calibration_source="manual_mpp",
        pixel_size_x_m=0.1, pixel_size_y_m=0.1,
    )
    poly_px = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    poly_m = pixel_polygon_to_working_crs(poly_px, geo_info, raster_transform=None)
    # 10px * 0.1 m/px = 1m per side -> 1m x 1m square
    xs = [p[0] for p in poly_m]
    ys = [p[1] for p in poly_m]
    assert math.isclose(max(xs) - min(xs), 1.0, rel_tol=1e-9)
    assert math.isclose(max(ys) - min(ys), 1.0, rel_tol=1e-9)
