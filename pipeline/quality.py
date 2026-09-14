"""Good / Moderate / Limited quality indicator.

Built ONLY from observable, defensible factors — never a fabricated
probability-of-correctness. Every label comes with the specific reasons
that produced it, so the UI can show "why", not just a colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QualityFactors:
    has_georeference: bool
    resolution_m_per_px: Optional[float]
    pct_detections_near_edge: float  # 0-100
    median_confidence: Optional[float]  # 0-1, None if no detections
    pct_low_confidence: float  # share of raw detections below 0.3, 0-100
    crown_overlap_ratio: Optional[float]  # union_area / naive_summed_area, 0-1 (lower = more overlap)


@dataclass
class QualityResult:
    label: str  # "Good" | "Moderate" | "Limited"
    reasons: List[str] = field(default_factory=list)


# Thresholds are intentionally simple and documented so they can be
# challenged/tuned — this is a rule-based checklist, not a learned score.
RESOLUTION_GOOD_M = 0.15   # DeepForest's NEON training imagery is ~0.1 m/px
RESOLUTION_LIMITED_M = 0.5
EDGE_GOOD_PCT = 10.0
EDGE_LIMITED_PCT = 30.0
CONF_GOOD = 0.5
CONF_LIMITED = 0.3
LOW_CONF_LIMITED_PCT = 40.0
OVERLAP_LIMITED_RATIO = 0.5  # union/sum below this = crowns are heavily overlapping/merged


def assess_quality(f: QualityFactors) -> QualityResult:
    limited_reasons: List[str] = []
    moderate_reasons: List[str] = []
    good_reasons: List[str] = []

    if not f.has_georeference:
        limited_reasons.append("No georeferencing — area/cover figures are unavailable or rely on a manual scale.")
    else:
        good_reasons.append("Image is georeferenced with a usable CRS.")

    if f.resolution_m_per_px is not None:
        if f.resolution_m_per_px > RESOLUTION_LIMITED_M:
            limited_reasons.append(
                f"Coarse resolution (~{f.resolution_m_per_px:.2f} m/px) — well beyond the "
                f"~0.1 m/px imagery DeepForest was trained on; small/adjacent crowns are likely missed or merged."
            )
        elif f.resolution_m_per_px > RESOLUTION_GOOD_M:
            moderate_reasons.append(
                f"Moderate resolution (~{f.resolution_m_per_px:.2f} m/px), coarser than the model's training imagery."
            )
        else:
            good_reasons.append(f"Resolution (~{f.resolution_m_per_px:.2f} m/px) is close to the model's training domain.")

    if f.pct_detections_near_edge > EDGE_LIMITED_PCT:
        limited_reasons.append(
            f"{f.pct_detections_near_edge:.0f}% of detections sit near a tile/image edge — "
            "elevated risk of duplicated or clipped crowns."
        )
    elif f.pct_detections_near_edge > EDGE_GOOD_PCT:
        moderate_reasons.append(f"{f.pct_detections_near_edge:.0f}% of detections are near a tile/image edge.")
    else:
        good_reasons.append("Few detections sit near a tile/image edge.")

    if f.median_confidence is not None:
        if f.median_confidence < CONF_LIMITED:
            limited_reasons.append(f"Low median detection confidence ({f.median_confidence:.2f}).")
        elif f.median_confidence < CONF_GOOD:
            moderate_reasons.append(f"Middling median detection confidence ({f.median_confidence:.2f}).")
        else:
            good_reasons.append(f"High median detection confidence ({f.median_confidence:.2f}).")

    if f.pct_low_confidence > LOW_CONF_LIMITED_PCT:
        limited_reasons.append(
            f"{f.pct_low_confidence:.0f}% of raw detections score below 0.3 — many candidates are marginal."
        )

    if f.crown_overlap_ratio is not None and f.crown_overlap_ratio < OVERLAP_LIMITED_RATIO:
        moderate_reasons.append(
            f"Crowns overlap heavily (union is only {f.crown_overlap_ratio * 100:.0f}% of summed individual areas) — "
            "dense/overlapping canopy is harder to delineate reliably."
        )

    if limited_reasons:
        return QualityResult(label="Limited", reasons=limited_reasons + moderate_reasons)
    if moderate_reasons:
        return QualityResult(label="Moderate", reasons=moderate_reasons + good_reasons)
    return QualityResult(label="Good", reasons=good_reasons or ["No risk factors detected."])
