from pipeline.tiling import compute_tile_grid, local_to_global, flag_near_edge


def test_single_tile_when_image_smaller_than_tile_size():
    grid = compute_tile_grid(width=300, height=200, tile_size=800, overlap=100)
    assert grid == [(0, 0, 300, 200)]


def test_full_coverage_no_gaps():
    width, height, tile_size, overlap = 2000, 1500, 800, 100
    grid = compute_tile_grid(width, height, tile_size, overlap)
    # Union of tile rectangles must cover every pixel of the image.
    covered = [[False] * width for _ in range(0)]  # not building full bool grid (memory); check via ranges instead
    x_ranges = sorted({(x, x + w) for (x, y, w, h) in grid})
    y_ranges = sorted({(y, y + h) for (x, y, w, h) in grid})

    def covers(ranges, extent):
        pts = set()
        for a, b in ranges:
            pts.update(range(a, b))
        return len(pts) == extent and min(pts) == 0 and max(pts) == extent - 1

    assert covers(x_ranges, width)
    assert covers(y_ranges, height)


def test_every_tile_is_full_size_not_partial():
    width, height, tile_size, overlap = 1950, 1030, 800, 100
    grid = compute_tile_grid(width, height, tile_size, overlap)
    for (x, y, w, h) in grid:
        assert w == min(tile_size, width)
        assert h == min(tile_size, height)
        assert x + w <= width
        assert y + h <= height
        assert x >= 0 and y >= 0


def test_adjacent_tiles_overlap_by_at_least_requested_amount():
    width, height, tile_size, overlap = 2400, 800, 800, 150
    grid = compute_tile_grid(width, height, tile_size, overlap)
    xs = sorted({x for (x, y, w, h) in grid})
    for a, b in zip(xs, xs[1:]):
        actual_overlap = (a + tile_size) - b
        assert actual_overlap >= overlap


def test_invalid_inputs_raise():
    import pytest
    with pytest.raises(ValueError):
        compute_tile_grid(100, 100, tile_size=0, overlap=0)
    with pytest.raises(ValueError):
        compute_tile_grid(100, 100, tile_size=50, overlap=50)  # overlap == tile_size
    with pytest.raises(ValueError):
        compute_tile_grid(0, 100, tile_size=50, overlap=10)


def test_local_to_global_offsets_and_clips():
    # tile at (100, 200) in a 500x500 image
    local_dets = [(10.0, 20.0, 50.0, 60.0, 0.9)]
    out = local_to_global(local_dets, x_off=100, y_off=200, tile_id=3, image_width=500, image_height=500)
    assert len(out) == 1
    d = out[0]
    assert d["xmin"] == 110.0
    assert d["ymin"] == 220.0
    assert d["xmax"] == 150.0
    assert d["ymax"] == 260.0
    assert d["tile_id"] == 3


def test_local_to_global_clips_to_image_bounds():
    # box extends past the image edge in global coords
    local_dets = [(700.0, 700.0, 900.0, 900.0, 0.5)]
    out = local_to_global(local_dets, x_off=0, y_off=0, tile_id=0, image_width=800, image_height=800)
    assert len(out) == 1
    assert out[0]["xmax"] == 800.0
    assert out[0]["ymax"] == 800.0


def test_local_to_global_drops_degenerate_boxes():
    # entirely outside the image -> clipped to a zero-area box -> dropped
    local_dets = [(-100.0, -100.0, -50.0, -50.0, 0.9)]
    out = local_to_global(local_dets, x_off=0, y_off=0, tile_id=0, image_width=500, image_height=500)
    assert out == []


def test_flag_near_edge():
    # detection right at tile's left border
    assert flag_near_edge(xmin=100, ymin=150, xmax=140, ymax=190, x_off=100, y_off=100, tile_w=400, tile_h=400, edge_margin_px=10) is True
    # detection safely in the middle
    assert flag_near_edge(xmin=250, ymin=250, xmax=290, ymax=290, x_off=100, y_off=100, tile_w=400, tile_h=400, edge_margin_px=10) is False
