"""Tree Crown Detection & Canopy Cover Estimation - Streamlit app.

Architecture note: `st.session_state.pipeline_result` is populated exactly
once per "Run Analysis" click (or once for the example, on first load).
The confidence slider below only ever calls `apply_threshold(...)` on that
cached result - it never re-runs detection. See pipeline/pipeline.py.
"""
from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import DetectorUnavailableError, try_build_default_detector
from pipeline.pipeline import apply_threshold, compute_quality, run_pipeline
from pipeline.image_io import ImageOpenError, ImageReader
from pipeline import geo as geo_mod
from pipeline import export as export_mod
from pipeline import validation as val_mod
from pipeline import precomputed as precomputed_mod
from app.overlay import draw_detections, draw_tile_grid, draw_ground_truth

EXAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "example", "OSBS_029.tif")
EXAMPLE_GT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "example", "OSBS_029.csv")
PRECOMPUTED_DIR = os.environ.get(
    "TREECROWN_PRECOMPUTED_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "precomputed"),
)

st.set_page_config(page_title="Tree Crown Detection & Canopy Cover", layout="wide")


# --------------------------------------------------------------------------
# Cached, expensive resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_detector():
    """Attempt to build the real detector exactly once per running app
    process. Returns (detector_or_None, error_message_or_None) - never
    raises, so a failed load is cached too (no repeated slow retries).
    """
    try:
        det, _msg = try_build_default_detector()
        return det, None
    except DetectorUnavailableError as e:
        return None, str(e)


def get_raster_transform(path):
    try:
        with ImageReader(path) as r:
            return r.transform
    except Exception:
        return None


# --------------------------------------------------------------------------
# Sidebar: image source
# --------------------------------------------------------------------------
st.title("🌳 Tree Crown Detection & Canopy Cover Estimation")
st.caption(
    "Detects individual tree crowns in aerial/satellite imagery and estimates canopy cover. "
    "Built for honesty over polish: every number below is labeled as measured or model-estimated."
)

with st.sidebar:
    st.header("1. Choose an image")
    use_example = st.button("▶ Try Example Dataset (judge mode)", width="stretch")
    uploaded = st.file_uploader("...or upload your own image", type=["tif", "tiff", "png", "jpg", "jpeg"])

    st.header("2. Detection settings")
    tile_size = st.number_input("Tile size (px)", min_value=200, max_value=2000, value=800, step=100)
    overlap = st.number_input("Tile overlap (px)", min_value=0, max_value=500, value=100, step=25)
    manual_mpp = None
    st.caption("If your image has no georeferencing, you can manually supply a scale:")
    manual_mpp_input = st.text_input("Manual scale (metres/pixel), optional", value="")
    if manual_mpp_input.strip():
        try:
            manual_mpp = float(manual_mpp_input)
        except ValueError:
            st.warning("Manual scale must be a number, ignoring it.")

    run_clicked = st.button("▶ Run Analysis", type="primary", width="stretch")


# --------------------------------------------------------------------------
# Resolve which image is active
# --------------------------------------------------------------------------
if use_example:
    st.session_state["image_path"] = EXAMPLE_IMAGE_PATH
    st.session_state["is_example"] = True
    st.session_state.pop("pipeline_result", None)

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1] or ".png"
    tmp_path = os.path.join(tempfile.gettempdir(), f"uploaded_{uploaded.name}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    if st.session_state.get("image_path") != tmp_path:
        st.session_state["image_path"] = tmp_path
        st.session_state["is_example"] = False
        st.session_state.pop("pipeline_result", None)

image_path = st.session_state.get("image_path")

if image_path is None:
    st.info("← Click **Try Example Dataset** in the sidebar for a one-click demo, or upload your own image.")
    st.stop()


# --------------------------------------------------------------------------
# Pre-flight: image + georeferencing info (works even without a detector)
# --------------------------------------------------------------------------
try:
    reader = ImageReader(image_path)
except ImageOpenError as e:
    st.error(f"Could not open this image: {e}")
    st.stop()

with reader:
    width, height = reader.width, reader.height
    src_transform = reader.transform

geo_info = geo_mod.build_geo_info(image_path, manual_mpp=manual_mpp)

if st.session_state.get("is_example"):
    st.info(
        "📌 **Bundled example dataset — for demonstration/pipeline validation only.** "
        "This is NOT the hackathon's target imagery and results here are not representative "
        "of performance on your actual imagery."
    )

col_img, col_info = st.columns([2, 1])
with col_img:
    preview = Image.open(image_path) if image_path.lower().endswith((".png", ".jpg", ".jpeg")) else None
    if preview is None:
        with ImageReader(image_path) as r:
            arr = r.read_window(0, 0, width, height)
        preview = Image.fromarray(arr)
    st.image(preview, caption=f"Input image ({width}×{height}px)", width="stretch")

with col_info:
    st.subheader("Georeferencing")
    if geo_info.has_crs:
        st.success(f"CRS detected: {geo_info.source_crs}")
        st.write(f"Resolution: **{geo_info.pixel_size_x_m:.3f} m/px** × {geo_info.pixel_size_y_m:.3f} m/px")
    elif geo_info.area_available:
        st.warning(f"No embedded CRS — using manual scale: **{geo_info.pixel_size_x_m} m/px**")
    else:
        st.error("No georeferencing found. Area/hectare figures are unavailable (canopy cover % is still shown).")
    for n in geo_info.notes:
        st.caption(n)


# --------------------------------------------------------------------------
# Detector status
# --------------------------------------------------------------------------
detector, detector_error = get_detector()

if detector_error:
    st.error(
        "**Detection engine unavailable in this environment.**\n\n"
        f"Reason: {detector_error}\n\n"
        "Everything else on this page (image loading, tiling plan, georeferencing math, "
        "and — once run — the confidence slider, exports, and validation scoring) is fully "
        "implemented and unit-tested. Only live model inference is blocked here."
    )

if run_clicked or (use_example and "pipeline_result" not in st.session_state):
    if detector is None:
        st.warning("Cannot run analysis: no working detector is available (see error above).")
    else:
        with st.spinner("Tiling image and running detection..."):
            try:
                result = run_pipeline(
                    image_path, detector,
                    tile_size=int(tile_size), overlap=int(overlap), manual_mpp=manual_mpp,
                )
                st.session_state["pipeline_result"] = result
                st.session_state["raster_transform"] = src_transform
            except ImageOpenError as e:
                st.error(f"Could not process this image: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Analysis failed with an unexpected error: {type(e).__name__}: {e}")
                st.session_state["live_inference_failed"] = True

# --------------------------------------------------------------------------
# Demo-failure safety net: ONLY offered explicitly, NEVER auto-substituted.
# Triggered when either (a) there's no working detector at all, or
# (b) a working detector exists but the live run just failed above.
# --------------------------------------------------------------------------
need_fallback_offer = (
    "pipeline_result" not in st.session_state
    and (detector is None or st.session_state.get("live_inference_failed"))
)
if need_fallback_offer:
    fallback_path = precomputed_mod.find_precomputed_fallback(image_path, PRECOMPUTED_DIR)
    if fallback_path:
        st.warning(
            "A precomputed result captured earlier is available for this image. "
            "This is **NOT a live run** - only load it if you understand that."
        )
        if st.button("⚠️ Load PRECOMPUTED results (not live)"):
            loaded = precomputed_mod.load_precomputed(fallback_path)
            # Matched by basename already (see find_precomputed_fallback) -
            # force image_path to the CURRENTLY active path so the
            # `result.image_path == image_path` check below is never
            # fooled by relative/absolute path string differences between
            # however the fallback file was originally created and how
            # this session referenced the image.
            loaded.image_path = image_path
            st.session_state["pipeline_result"] = loaded
            st.session_state["raster_transform"] = src_transform
            # No st.rerun() needed - the Results section below reads
            # session_state directly, later in this same script pass.
    else:
        st.caption("No precomputed fallback is available for this image either.")


# --------------------------------------------------------------------------
# Tiling preview (always available, even pre-detection)
# --------------------------------------------------------------------------
with st.expander("Tiling plan (before running detection)"):
    from pipeline import tiling as tiling_mod
    try:
        grid = tiling_mod.compute_tile_grid(width, height, int(tile_size), int(overlap))
        st.write(f"This image will be split into **{len(grid)} tile(s)** of {tile_size}×{tile_size}px "
                 f"with {overlap}px overlap, then merged back with cross-tile duplicate removal.")
        if len(grid) > 1:
            grid_img = draw_tile_grid(preview, grid)
            st.image(grid_img, width="stretch")
    except ValueError as e:
        st.warning(f"Tiling configuration issue: {e}")


# --------------------------------------------------------------------------
# Results (only if we have a cached PipelineResult)
# --------------------------------------------------------------------------
result = st.session_state.get("pipeline_result")

if result is not None and result.image_path == image_path:
    st.divider()
    st.subheader("Results")

    if result.used_precomputed_fallback:
        st.error(
            "🔴 **PRECOMPUTED — NOT LIVE.** These results were captured in an earlier run, "
            "not produced by this session. Live inference was unavailable when this was "
            "loaded (see the error above)."
        )

    threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01,
        help="Changing this re-filters the already-computed detections instantly — it never re-runs the model.",
    )
    view = apply_threshold(result, threshold, raster_transform=st.session_state.get("raster_transform"))
    quality_result = compute_quality(result, view)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tree crowns detected", view.count)
    if view.union_area_m2 is not None:
        m2.metric("Total canopy area", f"{view.union_area_m2:,.1f} m²", help=f"= {view.union_area_m2/10000:,.3f} ha")
    else:
        m2.metric("Total canopy area", "unavailable", help="No calibrated scale for this image.")
    if view.canopy_cover_pct is not None:
        m3.metric("Canopy cover", f"{view.canopy_cover_pct:.1f}%")
    else:
        m3.metric("Canopy cover", "n/a")
    badge = {"Good": "🟢", "Moderate": "🟡", "Limited": "🔴"}[quality_result.label]
    m4.metric("Quality signal", f"{badge} {quality_result.label}")

    with st.expander(f"Why this quality label? ({quality_result.label})", expanded=(quality_result.label != "Good")):
        for r in quality_result.reasons:
            st.write(f"- {r}")

    st.image(
        draw_detections(preview, view.detections),
        caption=f"{view.count} crowns shown at threshold ≥ {threshold:.2f} (color: red=low confidence → green=high)",
        width="stretch",
    )

    st.caption(
        f"Model: {result.detector_name}"
        + (" — PRECOMPUTED, NOT LIVE" if result.used_precomputed_fallback else "")
        + f". Raw candidates before thresholding: {len(result.raw_detections)}."
    )
    if result.warnings:
        with st.expander("Warnings from this run"):
            for w in result.warnings:
                st.write(f"- {w}")

    # ---- Exports ----
    st.subheader("Downloads")
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "detections.csv")
    geojson_path = os.path.join(tmp_dir, "detections.geojson")
    _data_source = "PRECOMPUTED — NOT LIVE" if result.used_precomputed_fallback else "LIVE INFERENCE"
    export_mod.write_csv(csv_path, view.detections, result.geo, data_source=_data_source)
    export_mod.write_geojson(geojson_path, view.detections, result.geo, st.session_state.get("raster_transform"), data_source=_data_source)

    dl1, dl2 = st.columns(2)
    with open(csv_path, "rb") as f:
        dl1.download_button("⬇ Download CSV", f, file_name="tree_crown_detections.csv", width="stretch")
    with open(geojson_path, "rb") as f:
        dl2.download_button("⬇ Download GeoJSON", f, file_name="tree_crown_detections.geojson", width="stretch")

    # ---- Validation (example dataset only, for now) ----
    if st.session_state.get("is_example"):
        st.divider()
        st.subheader("Validation against a reference-annotated patch")
        st.warning(
            "📌 **Bundled example dataset — for demonstration/pipeline validation only.** "
            "These reference boxes ship with the DeepForest package itself (a standard NEON "
            "tutorial tile), not hand-drawn for this project, and may overlap with the model's "
            "own training/benchmark data. **This is an implementation sanity-check, not "
            "validation on the hackathon's target imagery** — that imagery was not available "
            "when this was built."
        )
        gt_boxes = val_mod.load_ground_truth_csv(EXAMPLE_GT_CSV, image_filename_filter="OSBS_029.tif")
        preds_tuples = [(d.xmin, d.ymin, d.xmax, d.ymax, d.score) for d in view.detections]
        vr = val_mod.evaluate(gt_boxes, preds_tuples, iou_threshold=0.5)

        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Reference trees", vr.reference_count)
        v2.metric("Predicted", vr.predicted_count)
        v3.metric("Precision", f"{vr.precision:.2f}" if vr.precision is not None else "n/a")
        v4.metric("Recall", f"{vr.recall:.2f}" if vr.recall is not None else "n/a")
        v5.metric("F1", f"{vr.f1:.2f}" if vr.f1 is not None else "n/a")
        if vr.mean_iou_matched is not None:
            st.write(f"Mean IoU of matched boxes: **{vr.mean_iou_matched:.2f}**")

        st.image(draw_ground_truth(preview, gt_boxes), caption="Reference (ground-truth) boxes", width=350)

else:
    st.info("Click **Run Analysis** in the sidebar to detect trees on this image.")


# --------------------------------------------------------------------------
# Persistent "What should you trust?" section
# --------------------------------------------------------------------------
st.divider()
with st.expander("📋 What should you trust?", expanded=True):
    st.markdown(
        """
**Measured directly from the input data** (not model output):
- Image dimensions, and — when the image is georeferenced — the real ground resolution (m/px) and analyzed area.
- Canopy cover **%** is a directly measured ratio of detected-crown pixels to image pixels — it does not
  require georeferencing to be correct, only the absolute area (m²/hectares) does.

**Model estimates** (uncertain, not ground truth):
- Every crown box, the crown count, and each confidence score come from a pretrained detector
  (DeepForest / RetinaNet) trained primarily on NEON airborne forest imagery in the United States.
  Performance on imagery from a different ecosystem, sensor, or resolution is **unverified** until
  checked against real ground truth for that imagery.

**Main factors that reduce reliability:**
- Resolution far from the model's ~0.1 m/pixel training imagery.
- No georeferencing (blocks absolute area; cover % is still shown).
- A high share of detections sitting at a tile or image edge (duplication/clipping risk).
- Low or widely-spread confidence scores.
- Dense, heavily overlapping canopy, where individual crowns are hard to separate.

**This is not a substitute for field validation.** For carbon-market-grade or other high-stakes
decisions, treat every number here as a starting estimate to be checked against ground truth for
your specific site, not as verified fact.
        """
    )

st.caption(
    "Canopy area is computed from the UNION of detected crown polygons (overlaps are not double-counted). "
    "Changing the confidence threshold only re-filters already-computed detections — it never re-runs the model."
)
