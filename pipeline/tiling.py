"""Tile a large image into overlapping windows and map local detections
back to global image pixel coordinates.

Design choice: every tile is exactly `tile_size` x `tile_size` (never a
smaller partial tile at the image edge). Edge tiles are shifted inward so
they still fit within the image, which means edge tiles overlap their
neighbour by MORE than `overlap` px — that's fine, dedup (NMS) handles it.
This keeps the detector's input shape constant, which matters for models
that expect a fixed tile size.
"""
from __future__ import annotations

from typing import List, Tuple


def compute_tile_grid(
    width: int, height: int, tile_size: int, overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x_off, y_off, tile_w, tile_h) covering the image.

    Raises ValueError on nonsensical inputs so callers fail loudly instead
    of silently producing an empty/garbage grid.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and < tile_size")
    if width <= 0 or height <= 0:
        raise ValueError("image width/height must be positive")

    step = tile_size - overlap

    if width <= tile_size and height <= tile_size:
        # Whole image fits in a single tile - use the image's own size so we
        # don't request out-of-bounds pixels from a tiny image.
        return [(0, 0, width, height)]

    def positions(extent: int) -> List[int]:
        if extent <= tile_size:
            return [0]
        pos = list(range(0, extent - tile_size + 1, step))
        last = extent - tile_size
        if pos[-1] != last:
            pos.append(last)  # shift final tile inward to cover the remainder
        return pos

    xs = positions(width)
    ys = positions(height)

    grid = []
    for y in ys:
        th = min(tile_size, height)
        for x in xs:
            tw = min(tile_size, width)
            grid.append((x, y, tw, th))
    return grid


def local_to_global(
    detections_local: List[Tuple[float, float, float, float, float]],
    x_off: int,
    y_off: int,
    tile_id: int,
    image_width: int,
    image_height: int,
):
    """Convert (xmin, ymin, xmax, ymax, score) in tile-local pixel coords to
    global Detection-ready tuples, clipped to image bounds.

    Returns list of dicts (not Detection objects, to keep this module free
    of the schema import cycle / for easy unit testing).
    """
    out = []
    for (xmin, ymin, xmax, ymax, score) in detections_local:
        gxmin = max(0.0, min(float(image_width), x_off + xmin))
        gymin = max(0.0, min(float(image_height), y_off + ymin))
        gxmax = max(0.0, min(float(image_width), x_off + xmax))
        gymax = max(0.0, min(float(image_height), y_off + ymax))
        if gxmax <= gxmin or gymax <= gymin:
            continue  # degenerate box after clipping, drop it
        out.append(
            {
                "xmin": gxmin,
                "ymin": gymin,
                "xmax": gxmax,
                "ymax": gymax,
                "score": score,
                "tile_id": tile_id,
            }
        )
    return out


def flag_near_edge(
    xmin: float, ymin: float, xmax: float, ymax: float,
    x_off: int, y_off: int, tile_w: int, tile_h: int,
    edge_margin_px: float = 10.0,
) -> bool:
    """True if a detection sits within `edge_margin_px` of its own tile's
    border (a useful signal for the quality indicator — detections right at
    a tile seam are the ones most likely to be duplicated/clipped)."""
    local_xmin = xmin - x_off
    local_ymin = ymin - y_off
    local_xmax = xmax - x_off
    local_ymax = ymax - y_off
    return (
        local_xmin <= edge_margin_px
        or local_ymin <= edge_margin_px
        or (tile_w - local_xmax) <= edge_margin_px
        or (tile_h - local_ymax) <= edge_margin_px
    )
