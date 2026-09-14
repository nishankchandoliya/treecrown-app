"""Greedy NMS to merge duplicate detections produced when the same tree
crown falls inside more than one overlapping tile.
"""
from __future__ import annotations

from typing import List
from pipeline.schema import Detection


def box_iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.xmin, b.xmin)
    iy1 = max(a.ymin, b.ymin)
    ix2 = min(a.xmax, b.xmax)
    iy2 = min(a.ymax, b.ymax)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.xmax - a.xmin) * max(0.0, a.ymax - a.ymin)
    area_b = max(0.0, b.xmax - b.xmin) * max(0.0, b.ymax - b.ymin)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_duplicates(detections: List[Detection], iou_threshold: float = 0.4) -> List[Detection]:
    """Greedy NMS across ALL detections (regardless of which tile produced
    them). Highest-score detection wins each cluster; suppressed boxes
    (same crown, seen from a neighbouring tile) are dropped entirely.

    This runs once on the RAW detection set (all confidence scores) so that
    later confidence filtering never needs to re-run merging.
    """
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    suppressed = [False] * len(ordered)

    for i, det in enumerate(ordered):
        if suppressed[i]:
            continue
        kept.append(det)
        for j in range(i + 1, len(ordered)):
            if suppressed[j]:
                continue
            if box_iou(det, ordered[j]) >= iou_threshold:
                suppressed[j] = True

    return kept
