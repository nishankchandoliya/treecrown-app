"""Draws detection boxes (and, optionally, tile grid lines) on top of the
source image for visual sanity-checking in the UI."""
from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from pipeline.schema import Detection


def score_to_color(score: float) -> Tuple[int, int, int]:
    """Low confidence -> orange/red, high confidence -> green."""
    score = max(0.0, min(1.0, score))
    r = int(255 * (1 - score))
    g = int(255 * score)
    return (r, g, 40)


def draw_detections(
    base_image: Image.Image,
    detections: List[Detection],
    color_by_confidence: bool = True,
    box_width: int = 2,
) -> Image.Image:
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        color = score_to_color(d.score) if color_by_confidence else (255, 0, 0)
        draw.rectangle([d.xmin, d.ymin, d.xmax, d.ymax], outline=color, width=box_width)
    return img


def draw_tile_grid(
    base_image: Image.Image,
    tile_grid: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (60, 140, 255),
) -> Image.Image:
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in tile_grid:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
    return img


def draw_ground_truth(
    base_image: Image.Image,
    gt_boxes,
    color: Tuple[int, int, int] = (255, 0, 255),
) -> Image.Image:
    img = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for b in gt_boxes:
        draw.rectangle([b.xmin, b.ymin, b.xmax, b.ymax], outline=color, width=2)
    return img
