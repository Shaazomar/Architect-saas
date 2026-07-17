"""Stage 2 — structural element detection.

MVP implementation: walls are the thick strokes of the drawing, so a
morphological opening sized from the estimated stroke thickness removes text,
dimension lines and furniture symbols while keeping walls intact.

This stage is the primary ML plug-in point: a YOLO/RT-DETR + SAM2 detector
should return the same ``DetectedStructure`` contract.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import PlanImageError
from .types import DetectedStructure, PreprocessedPlan


def _estimate_wall_thickness(ink: np.ndarray) -> float:
    """Dominant stroke thickness = 2x the distance-transform peak values.

    Thin annotation strokes dominate by count, so we look at the upper tail of
    the distance distribution, which the wall interiors produce.
    """
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    inside = dist[dist > 0.5]
    if inside.size == 0:
        raise PlanImageError("No structural linework found in the image.")
    thickness = 2.0 * float(np.percentile(inside, 92))
    return float(np.clip(thickness, 3.0, 60.0))


def detect_structure(plan: PreprocessedPlan) -> DetectedStructure:
    thickness = _estimate_wall_thickness(plan.ink)

    # Opening with a kernel just under the wall thickness erases everything
    # thinner than a wall (text, hatching, dimension lines).
    k = max(3, int(round(thickness * 0.55)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    walls = cv2.morphologyEx(plan.ink, cv2.MORPH_OPEN, kernel)

    # Re-join walls the opening may have nicked at junctions, without sealing
    # door openings (kernel stays below typical door widths).
    closing = cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2 * 2 + 1, k // 2 * 2 + 1))
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, closing)

    if cv2.countNonZero(walls) == 0:
        raise PlanImageError("No walls could be detected in the drawing.")

    return DetectedStructure(wall_mask=walls, wall_thickness_px=thickness)
