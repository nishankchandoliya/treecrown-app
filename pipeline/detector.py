"""Detector interface.

IMPORTANT: this module must import cleanly even when torch/deepforest are
broken or unavailable in the current environment. All heavy imports are
deferred into method bodies, not module top-level, so the rest of the app
(tiling, geo math, UI) never breaks because of the model stack.

There is deliberately NO synthetic/fake detector in this file. A detector
that invents tree boxes belongs only in tests/, imported only by tests, so
it can never accidentally reach the real app / a judge-facing demo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class DetectorUnavailableError(RuntimeError):
    """Raised when a real detector cannot be constructed or run in this
    environment. Carries a clear, specific reason so the caller (app UI)
    can show an honest error instead of pretending inference happened."""


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def predict_tile(self, image_rgb) -> List[Tuple[float, float, float, float, float]]:
        """Run detection on a single tile (H, W, 3) RGB uint8 array.

        Returns list of (xmin, ymin, xmax, ymax, score) in TILE-LOCAL pixel
        coordinates.
        """
        raise NotImplementedError


class DeepForestDetector(BaseDetector):
    """Wraps weecology/DeepForest's pretrained RetinaNet crown detector.

    Loading is lazy and explicit: nothing happens at import time or even
    at __init__ time beyond storing config. The actual model load (which
    is what requires network access to Hugging Face Hub) happens on first
    call to `predict_tile`, wrapped so failures come back as a
    DetectorUnavailableError with a specific, honest reason rather than a
    raw stack trace or — worse — a silent fallback to fake output.
    """

    name = "DeepForest (pretrained RetinaNet)"

    def __init__(self, model_repo: str = "weecology/deepforest-tree"):
        self.model_repo = model_repo
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise DetectorUnavailableError(
                "PyTorch is not usable in this environment "
                f"({type(e).__name__}: {e}). DeepForest requires a working "
                "torch install."
            ) from e

        try:
            from deepforest import main as df_main
        except Exception as e:  # noqa: BLE001
            raise DetectorUnavailableError(
                f"Could not import deepforest ({type(e).__name__}: {e})."
            ) from e

        try:
            model = df_main.deepforest()
            model.load_model(model_name=self.model_repo)
            model.eval()
        except Exception as e:  # noqa: BLE001
            raise DetectorUnavailableError(
                "Could not load DeepForest's pretrained weights "
                f"({type(e).__name__}: {e}). This model is hosted on the "
                "Hugging Face Hub (weecology/deepforest-tree) — this "
                "environment may not have network access to huggingface.co, "
                "or lacks the weight file locally."
            ) from e

        self._model = model

    def predict_tile(self, image_rgb) -> List[Tuple[float, float, float, float, float]]:
        self._ensure_loaded()
        import pandas as pd  # local import, cheap, already a dependency

        result = self._model.predict_image(image=image_rgb)
        if result is None or len(result) == 0:
            return []
        out = []
        for _, row in result.iterrows():
            out.append(
                (
                    float(row["xmin"]),
                    float(row["ymin"]),
                    float(row["xmax"]),
                    float(row["ymax"]),
                    float(row.get("score", row.get("scores", 1.0))),
                )
            )
        return out


def try_build_default_detector() -> Tuple[BaseDetector, str]:
    """Attempt to construct the real detector and confirm it actually loads.

    Returns (detector, status_message). Raises DetectorUnavailableError if
    it genuinely cannot be used, with a message good enough to show a user.
    """
    det = DeepForestDetector()
    det._ensure_loaded()  # force the load now, fail fast with a clear reason
    return det, f"{det.name} loaded successfully."
