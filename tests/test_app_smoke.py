"""Headless smoke tests for app/app.py using Streamlit's AppTest framework.

No real browser needed - these actually execute the app's script logic and
catch runtime errors that a pure syntax check would miss (missing session
state keys, bad widget wiring, path-matching bugs, etc).
"""
import os

import pytest

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "app.py")


def _fresh_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP_PATH)


def test_initial_load_has_no_exceptions():
    at = _fresh_app()
    at.run(timeout=60)
    assert len(at.exception) == 0


def test_try_example_with_no_detector_shows_honest_error_not_a_crash():
    """In this dev sandbox there is no working detector - the app must
    degrade honestly (clear error message), never crash and never fake a
    result."""
    at = _fresh_app()
    at.run(timeout=60)
    at.button[0].click()  # "Try Example Dataset"
    at.run(timeout=60)

    assert len(at.exception) == 0
    assert len(at.error) >= 1
    assert any("unavailable" in e.value.lower() for e in at.error)
    # Must NOT show any result metrics when there's no real detection.
    assert len(at.metric) == 0


def test_what_should_you_trust_section_always_present():
    at = _fresh_app()
    at.run(timeout=60)
    at.button[0].click()
    at.run(timeout=60)
    labels = [exp.label for exp in at.expander]
    assert any("what should you trust" in l.lower() for l in labels)


def test_precomputed_fallback_flow_end_to_end(tmp_path, monkeypatch):
    """The core demo-failure safety net: when no live detector is available
    but a precomputed result exists for this exact image, the app must
    offer it explicitly (never auto-load), and once loaded, must label it
    as precomputed everywhere - never presented as a live result.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.pipeline import run_pipeline
    from pipeline import precomputed as precomputed_mod
    from tests.fixtures import FakeDetector

    example_image = os.path.join(os.path.dirname(__file__), "..", "data", "example", "OSBS_029.tif")

    # Build a throwaway result purely to populate the test fallback file -
    # NOT real model output, never written to the real data/precomputed/.
    fake_detector = FakeDetector(boxes_fn=lambda t: [(50, 50, 90, 90, 0.8), (200, 200, 240, 240, 0.6)])
    result = run_pipeline(example_image, fake_detector, tile_size=800, overlap=100)
    fallback_path = precomputed_mod.precomputed_path_for_image(example_image, str(tmp_path))
    precomputed_mod.save_precomputed(result, fallback_path)

    monkeypatch.setenv("TREECROWN_PRECOMPUTED_DIR", str(tmp_path))

    at = _fresh_app()
    at.run(timeout=60)
    at.button[0].click()  # Try Example Dataset
    at.run(timeout=60)

    assert len(at.exception) == 0
    # Fallback must be OFFERED, not auto-applied - no results yet.
    assert len(at.metric) == 0
    fallback_buttons = [b for b in at.button if "PRECOMPUTED" in b.label]
    assert len(fallback_buttons) == 1

    fallback_buttons[0].click()
    at.run(timeout=60)

    assert len(at.exception) == 0
    metrics = {m.label: m.value for m in at.metric}
    assert "Tree crowns detected" in metrics
    # Must be unmistakably labeled as precomputed, not live - the exact
    # required phrasing, not just a paraphrase.
    assert any("precomputed" in e.value.lower() and "not live" in e.value.lower() for e in at.error)
    assert len(at.download_button) == 2


def test_app_source_uses_the_exact_required_precomputed_stamp():
    """Static guard: the app must use the literal phrase 'PRECOMPUTED — NOT
    LIVE' for exports, not just a similar-sounding label, per the explicit
    requirement that this stamp appear unmistakably everywhere relevant."""
    with open(APP_PATH, encoding="utf-8") as f:
        source = f.read()
    assert "PRECOMPUTED — NOT LIVE" in source
    assert "for demonstration/pipeline validation only" in source
