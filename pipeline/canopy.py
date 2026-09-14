"""Union-based canopy area and cover.

Requirement this exists to satisfy: overlapping crown polygons must not be
double-counted. Total canopy area = area of the UNION of all crown
polygons, not the sum of each polygon's individual area.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


def _safe_polygon(coords: List[Tuple[float, float]]) -> Optional[Polygon]:
    if len(coords) < 4:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def union_area(polygons_coords: List[List[Tuple[float, float]]]) -> Tuple[float, float]:
    """Returns (union_area, naive_summed_area) in whatever units the input
    coordinates are in (expected: square metres, already-projected).

    naive_summed_area is returned ONLY for internal sanity-checking/tests
    (to prove union <= sum whenever there's overlap) — it must never be
    reported to the user as "total canopy area".
    """
    polys = []
    naive_sum = 0.0
    for coords in polygons_coords:
        p = _safe_polygon(coords)
        if p is None:
            continue
        polys.append(p)
        naive_sum += p.area

    if not polys:
        return 0.0, 0.0

    merged = unary_union(polys)
    return merged.area, naive_sum


def canopy_cover_pct(union_area_m2: float, analyzed_area_m2: float) -> Optional[float]:
    if analyzed_area_m2 is None or analyzed_area_m2 <= 0:
        return None
    pct = (union_area_m2 / analyzed_area_m2) * 100.0
    # Clamp for float noise only (crown polygons occasionally straddle the
    # analyzed-area boundary by a hair); anything grossly over 100 signals a
    # real bug and should NOT be silently clamped away.
    if 100.0 < pct <= 100.5:
        pct = 100.0
    return pct
