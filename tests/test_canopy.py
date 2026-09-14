import math
from pipeline.canopy import union_area, canopy_cover_pct


def test_non_overlapping_squares_union_equals_sum():
    sq1 = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    sq2 = [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)]
    u, s = union_area([sq1, sq2])
    assert math.isclose(u, 2.0, rel_tol=1e-9)
    assert math.isclose(s, 2.0, rel_tol=1e-9)
    assert math.isclose(u, s)


def test_overlapping_squares_union_less_than_sum():
    # two 1x1 unit squares overlapping by a 0.5 x 0.5 corner
    sq1 = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    sq2 = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]
    u, s = union_area([sq1, sq2])
    # naive sum would double-count the 0.25 overlap area
    assert math.isclose(s, 2.0, rel_tol=1e-9)
    assert math.isclose(u, 1.75, rel_tol=1e-9)
    assert u < s


def test_fully_overlapping_squares_union_equals_one_area_not_sum():
    sq = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    u, s = union_area([sq, sq])  # identical polygon twice (duplicate detection scenario)
    assert math.isclose(u, 4.0, rel_tol=1e-9)  # union = one crown's area
    assert math.isclose(s, 8.0, rel_tol=1e-9)  # naive sum double-counts
    assert u == 4.0 and s == 8.0


def test_empty_input():
    u, s = union_area([])
    assert u == 0.0 and s == 0.0


def test_degenerate_polygon_ignored():
    # a "polygon" with fewer than 3 distinct points shouldn't crash or contribute area
    bad = [(0, 0), (1, 0)]
    good = [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]
    u, s = union_area([bad, good])
    assert math.isclose(u, 1.0, rel_tol=1e-9)


def test_canopy_cover_pct_basic():
    assert math.isclose(canopy_cover_pct(50.0, 200.0), 25.0)


def test_canopy_cover_pct_handles_zero_analyzed_area():
    assert canopy_cover_pct(10.0, 0.0) is None


def test_canopy_cover_pct_is_scale_invariant():
    # Same ratio whether measured in pixel-space or metre-space (as long as
    # the same uniform scale is applied to both numerator and denominator) -
    # this justifies always showing cover% even without georeferencing.
    px_cover = canopy_cover_pct(500.0, 2000.0)      # pixel^2 units
    m_cover = canopy_cover_pct(500.0 * 0.01, 2000.0 * 0.01)  # scaled by (0.1 m/px)^2 = 0.01
    assert math.isclose(px_cover, m_cover, rel_tol=1e-9)
