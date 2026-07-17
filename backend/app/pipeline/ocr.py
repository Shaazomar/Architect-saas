"""OCR stage plug-in point.

Room names, dimension strings and the scale bar come from OCR (PaddleOCR /
TrOCR per the platform blueprint). Those engines are heavyweight optional
dependencies, so the pipeline depends only on this Protocol; the default
implementation returns nothing and the pipeline proceeds with inferred
labels and wall-thickness-derived scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class OcrResult:
    # (text, x, y, confidence) in pixel coordinates
    texts: list[tuple[str, float, float, float]] = field(default_factory=list)
    # meters-per-pixel if a scale annotation was read, else None
    meters_per_px: float | None = None


class OcrEngine(Protocol):
    def read(self, image: np.ndarray) -> OcrResult: ...


class NullOcr:
    """Default engine: no text extraction."""

    def read(self, image: np.ndarray) -> OcrResult:  # noqa: ARG002
        return OcrResult()
