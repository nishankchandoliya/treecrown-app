"""Test-only fixtures.

`FakeDetector` deliberately lives here, not in pipeline/, and is imported
ONLY by test modules. It exists purely to test tiling/dedup/threshold
mechanics without needing a real model. It must never be reachable from the
Streamlit app or any judge-facing code path.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from pipeline.detector import BaseDetector


class FakeDetector(BaseDetector):
    """Returns whatever `boxes_fn(tile_image) -> [(xmin,ymin,xmax,ymax,score),...]`
    produces, and counts how many times it was actually called — used to
    prove the confidence-threshold slider never re-triggers detection.
    """

    name = "FakeDetector (test-only, not a real model)"

    def __init__(self, boxes_fn: Callable = None, fixed_boxes: List[Tuple[float, float, float, float, float]] = None):
        self.call_count = 0
        self._boxes_fn = boxes_fn
        self._fixed_boxes = fixed_boxes or []

    def predict_tile(self, image_rgb):
        self.call_count += 1
        if self._boxes_fn is not None:
            return self._boxes_fn(image_rgb)
        return list(self._fixed_boxes)
