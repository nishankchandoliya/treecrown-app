from pipeline.quality import QualityFactors, assess_quality


def good_factors(**overrides):
    base = dict(
        has_georeference=True,
        resolution_m_per_px=0.1,
        pct_detections_near_edge=2.0,
        median_confidence=0.8,
        pct_low_confidence=5.0,
        crown_overlap_ratio=0.9,
    )
    base.update(overrides)
    return QualityFactors(**base)


def test_all_good_factors_yields_good():
    result = assess_quality(good_factors())
    assert result.label == "Good"
    assert len(result.reasons) > 0


def test_no_georeference_forces_limited():
    result = assess_quality(good_factors(has_georeference=False))
    assert result.label == "Limited"
    assert any("georef" in r.lower() for r in result.reasons)


def test_coarse_resolution_forces_limited():
    result = assess_quality(good_factors(resolution_m_per_px=2.0))
    assert result.label == "Limited"


def test_moderate_resolution_alone_is_moderate_not_limited():
    result = assess_quality(good_factors(resolution_m_per_px=0.3))
    assert result.label == "Moderate"


def test_high_edge_share_forces_limited():
    result = assess_quality(good_factors(pct_detections_near_edge=45.0))
    assert result.label == "Limited"


def test_low_confidence_forces_limited():
    result = assess_quality(good_factors(median_confidence=0.15))
    assert result.label == "Limited"


def test_reasons_are_specific_not_generic():
    result = assess_quality(good_factors(has_georeference=False, resolution_m_per_px=1.0))
    joined = " ".join(result.reasons)
    assert "georef" in joined.lower()
    assert "resolution" in joined.lower() or "m/px" in joined.lower()


def test_label_is_deterministic():
    f = good_factors()
    r1 = assess_quality(f)
    r2 = assess_quality(f)
    assert r1.label == r2.label
    assert r1.reasons == r2.reasons
