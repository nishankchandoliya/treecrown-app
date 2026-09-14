"""Unified image reading via rasterio/GDAL.

Confirmed empirically: GDAL opens plain PNG/JPG the same way it opens
GeoTIFF (crs simply comes back None for non-georeferenced files), and
supports windowed reads for both — so one code path handles every input
format the brief requires, and memory usage stays bounded to one tile at a
time regardless of total file size.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window


class ImageOpenError(RuntimeError):
    """Raised for corrupted/unreadable input files — caller should show a
    clear user-facing error, never crash."""


@dataclass
class ImageReader:
    path: str

    def __post_init__(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                self._src = rasterio.open(self.path)
        except Exception as e:  # noqa: BLE001
            raise ImageOpenError(f"Could not open image '{self.path}': {e}") from e

        if self._src.width <= 0 or self._src.height <= 0:
            self._src.close()
            raise ImageOpenError(f"Image '{self.path}' has invalid dimensions.")

    @property
    def width(self) -> int:
        return self._src.width

    @property
    def height(self) -> int:
        return self._src.height

    @property
    def crs(self):
        return self._src.crs

    @property
    def transform(self):
        return self._src.transform

    @property
    def band_count(self) -> int:
        return self._src.count

    def read_window(self, x_off: int, y_off: int, w: int, h: int) -> np.ndarray:
        """Returns an (H, W, 3) uint8 RGB array for the requested window,
        clipped to image bounds. Handles 1-band (grayscale->replicated to
        3 channels), 3-band, and 4-band (drops alpha) sources.
        """
        x_off = max(0, min(x_off, self.width - 1))
        y_off = max(0, min(y_off, self.height - 1))
        w = max(1, min(w, self.width - x_off))
        h = max(1, min(h, self.height - y_off))
        window = Window(x_off, y_off, w, h)

        data = self._src.read(window=window)  # (bands, H, W)
        bands = data.shape[0]

        if bands == 1:
            rgb = np.repeat(data, 3, axis=0)
        elif bands >= 3:
            rgb = data[:3]
        else:  # 2 bands - unusual; duplicate first band into 3
            rgb = np.stack([data[0], data[0], data[0]], axis=0)

        rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)

        if rgb.dtype != np.uint8:
            # Robust min-max stretch per-tile for non-8-bit source data
            # (some GeoTIFFs are 16-bit). Documented as a simplification —
            # a production system would use a fixed, image-wide stretch.
            lo, hi = np.percentile(rgb, [1, 99]) if rgb.size else (0, 1)
            if hi <= lo:
                hi = lo + 1
            rgb = np.clip((rgb.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(rgb)

    def close(self):
        self._src.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
