---
title: Tree Crown Detection & Canopy Cover
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.63.0"
app_file: app/app.py
pinned: false
---

# Tree Crown Detection & Canopy Cover Estimation

Detects individual tree crowns in aerial/satellite forest imagery, computes
union-based canopy area/cover with correct CRS handling, and presents
results with an explicit honesty layer (measured vs. model-estimated,
a rule-based quality indicator, and real validation numbers).

## Status
Core pipeline (tiling, cross-tile dedup, georeferencing/CRS math,
union-based canopy area & cover, confidence-threshold filtering, CSV/GeoJSON
export, quality indicator, validation scoring, precomputed-fallback safety
net) is implemented and covered by 75 unit/integration/app-smoke tests,
including a genuinely large (27-megapixel, 63-tile) image to prove the
tiling path is memory-safe, not just correct in isolated math tests. The
Streamlit app wires all of this together and is tested headlessly with
Streamlit's `AppTest` framework (no browser needed) - including the
demo-failure fallback flow end-to-end.

**Not yet runnable in this dev sandbox**: real DeepForest inference (needs
a working torch install + Hugging Face Hub access - see
`docs/DEV_NOTES.md` for exactly why, and why it's expected to work fine on
the actual deployment target). Everything downstream of "a detector exists"
has been verified using a test-only fake detector that never ships in the
real app.

**Known gaps, deferred on purpose (see brief's own priority order):**
- KML clipping/calibration - not started.
- Crown polygon refinement (segmentation beyond the bounding box) - boxes
  are used as the polygon proxy for area math, which is an honest,
  documented simplification, not a bug.
- Biomass/carbon proxy - explicitly lowest priority, not started.
- `data/precomputed/` exists but is currently empty - there's no real
  DeepForest output yet to seed the demo-failure fallback with. One-line
  fix once a working detector runs anywhere: `pipeline.precomputed.save_precomputed(result, path)`.

## Run locally
```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/streamlit run app/app.py
```

## Run tests
```bash
./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest tests/ -v
```

## Project layout
```
pipeline/       core logic (tiling, dedup, geo/CRS math, canopy area,
                detector interface, quality indicator, validation,
                precomputed-fallback, export)
app/            Streamlit UI
tests/          75 tests: unit tests (synthetic data + the real bundled
                example), a large-image integration test, and headless
                app smoke tests (Streamlit AppTest)
data/example/   OSBS_029.tif + annotations, extracted from the deepforest
                package's own tutorial data (see docs/DEV_NOTES.md)
data/precomputed/  demo-failure fallback results go here (empty for now)
docs/           dev notes, (write-up to come)
```
