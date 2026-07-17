"""Stage 1 — decode and binarize the uploaded drawing.

Decoding happens from bytes we already size-checked at the API boundary; the
pixel cap here is the second line of defense against decompression bombs.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import settings
from .types import PreprocessedPlan


class PlanImageError(ValueError):
    """The upload is not a usable plan image. Message is safe to show users."""


def preprocess(image_bytes: bytes) -> PreprocessedPlan:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise PlanImageError("File could not be decoded as an image.")

    h, w = img.shape[:2]
    if h * w > settings.max_pixels:
        raise PlanImageError("Image resolution exceeds the allowed maximum.")
    if min(h, w) < settings.min_image_side:
        raise PlanImageError("Image is too small to contain a floor plan.")

    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Plans are dark ink on light paper; Otsu picks the split point. THRESH_BINARY_INV
    # makes ink white (255) which is what the morphology stages expect.
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if cv2.countNonZero(ink) == 0:
        raise PlanImageError("Image appears to be blank.")

    return PreprocessedPlan(ink=ink, width=w, height=h)
