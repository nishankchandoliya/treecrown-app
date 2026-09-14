"""Compares manually-verified reference (ground-truth) tree boxes against
model detections: reference count, predicted count, precision, recall, F1,
and mean IoU over matched pairs.

Matching strategy: standard greedy detection-evaluation matching. Sort
predictions by confidence (highest first); for each prediction, assign it
to the highest-IoU still-unmatched ground-truth box IF that IoU clears
`iou_threshold`; otherwise it's a false positive. Any ground-truth box left
unmatched at the end is a false negative.

This module knows nothing about where ground truth comes from — it could
be a genuinely hand-drawn patch, or an existing annotation file the user
supplies for review. It is the caller's responsibility to be honest about
provenance (see README / UI copy) — this module just does the arithmetic
correctly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GTBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "Tree"


@dataclass
class MatchRecord:
    gt_index: Optional[int]
    pred_index: Optional[int]
    iou: Optional[float]
    kind: str  # "tp" | "fp" | "fn"


@dataclass
class ValidationResult:
    reference_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    mean_iou_matched: Optional[float]
    iou_threshold: float
    matches: List[MatchRecord] = field(default_factory=list)


def load_ground_truth_csv(path: str, image_filename_filter: Optional[str] = None) -> List[GTBox]:
    """Parses the standard `image_path,xmin,ymin,xmax,ymax,label` format
    (matches DeepForest's own annotation CSV convention, so files in that
    format can be dropped in directly).
    """
    boxes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if image_filename_filter is not None and row.get("image_path") != image_filename_filter:
                continue
            boxes.append(
                GTBox(
                    xmin=float(row["xmin"]), ymin=float(row["ymin"]),
                    xmax=float(row["xmax"]), ymax=float(row["ymax"]),
                    label=row.get("label", "Tree"),
                )
            )
    return boxes


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(
    ground_truth: List[GTBox],
    predictions: List[Tuple[float, float, float, float, float]],  # (xmin,ymin,xmax,ymax,score)
    iou_threshold: float = 0.5,
) -> ValidationResult:
    gt_boxes = [(g.xmin, g.ymin, g.xmax, g.ymax) for g in ground_truth]
    preds_sorted = sorted(range(len(predictions)), key=lambda i: predictions[i][4], reverse=True)

    matched_gt = set()
    matches: List[MatchRecord] = []
    ious_matched: List[float] = []
    tp = 0
    fp = 0

    for pi in preds_sorted:
        px1, py1, px2, py2, _score = predictions[pi]
        best_iou = 0.0
        best_gi = None
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = _iou((px1, py1, px2, py2), gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            tp += 1
            ious_matched.append(best_iou)
            matches.append(MatchRecord(gt_index=best_gi, pred_index=pi, iou=best_iou, kind="tp"))
        else:
            fp += 1
            matches.append(MatchRecord(gt_index=None, pred_index=pi, iou=best_iou if best_gi is not None else None, kind="fp"))

    fn = len(gt_boxes) - len(matched_gt)
    for gi in range(len(gt_boxes)):
        if gi not in matched_gt:
            matches.append(MatchRecord(gt_index=gi, pred_index=None, iou=None, kind="fn"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    mean_iou = sum(ious_matched) / len(ious_matched) if ious_matched else None

    return ValidationResult(
        reference_count=len(gt_boxes),
        predicted_count=len(predictions),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou_matched=mean_iou,
        iou_threshold=iou_threshold,
        matches=matches,
    )
