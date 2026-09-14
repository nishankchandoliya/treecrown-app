# Dev notes: why the detector can't run in the dev sandbox (and why that's OK)

This pipeline was built in a sandboxed CPU-only container with a restricted
network allowlist and ~10GB of disk. Two independent things block running
DeepForest's actual pretrained model *in that sandbox specifically*:

## 1. Disk + mandatory CUDA runtime libraries
The default PyPI Linux wheel for `torch` (as of the version resolved here,
2.14.0) dynamically links against CUDA runtime shared libraries
(`libcudart`, `libcublasLt`, `libnvrtc`, ...) even for pure CPU tensor ops -
`import torch` fails outright without them. Pulling the full dependency
closure (`nvidia-cublas`, `nvidia-cudnn-cu13`, `nvidia-cusolver`, `triton`,
...) needs several GB, which exceeded the sandbox's disk quota. The actual
CPU-only wheel (no CUDA deps, ~200MB) is only published under PyTorch's own
package index (`download.pytorch.org`), which is not on this sandbox's
network allowlist - only PyPI/npm/GitHub domains are.

## 2. Hugging Face Hub access
DeepForest's `load_model()` downloads pretrained weights from the Hugging
Face Hub (`weecology/deepforest-tree`). `huggingface.co` is also not on the
sandbox's network allowlist.

## Why this doesn't block the actual deliverable
Neither of these is expected to be a problem on the real deployment target:
- **Hugging Face Spaces** / **Streamlit Community Cloud** both have normal,
  unrestricted internet access and a normal amount of disk, so
  `pip install -r requirements.txt` should pull a working torch + DeepForest
  without any of the workarounds attempted here.
- Everything that does NOT require the model itself - tiling, cross-tile
  dedup, CRS-correct area/cover math, the export formats, the quality
  indicator, the validation scorer, and the full Streamlit UI - has been
  built and unit-tested (59 tests) against this exact code, using a
  test-only fake detector (`tests/fixtures.py::FakeDetector`) that is
  never imported by the real app.

## What to do once you have a working detector locally (or just deploy)
1. `pip install -r requirements.txt` somewhere with normal internet access.
2. `streamlit run app/app.py`, click "Try Example Dataset", click "Run
   Analysis".
3. If it works, `pipeline/detector.py`'s `DeepForestDetector` is already
   wired up - no code changes needed.
4. If DeepForest still can't reach huggingface.co in your environment,
   download the `weecology/deepforest-tree` weights manually and point
   `DeepForestDetector(model_repo=...)` at a local path instead (small code
   change - ask and I'll make it if that's the path we need).

## The OSBS_029 example/validation dataset
`data/example/OSBS_029.tif` + `OSBS_029.csv` were extracted directly from
the `deepforest` PyPI package's own bundled sample/tutorial data (a
400x400px, 0.1m/px NEON tile, EPSG:32617, with 61 pre-existing tree
bounding-box annotations). This is **not** a patch we hand-annotated, and
it may overlap with DeepForest's own training/benchmark data - so any
precision/recall/F1 numbers computed against it are an implementation
sanity-check for the scoring code, not evidence of real-world accuracy.
Real validation for the submission should come from a patch drawn from the
actual target imagery, per the brief.
