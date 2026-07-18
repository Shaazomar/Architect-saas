"""Openings detection — doors and passages, straight from the drawing.

The room-segmentation stage seals wall gaps morphologically; the difference
between the sealed mask and the raw wall mask is therefore *exactly* the
material that was added to bridge each opening. Every connected component of
that difference which touches two rooms (or a room and the exterior) is an
opening. Nothing is invented: an opening exists only where the drawing has a
gap in a wall.

Swing direction needs the door-leaf arc symbol, which is thin linework — that
lands with the ML detector stage and is reported as null until then.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import DetectedStructure, PreprocessedPlan, VectorPlan
from .vectorize import seal_walls

# Plausible clear widths for an opening, in meters. Components outside this
# range are sealing artifacts (corner fills), not openings.
_MIN_WIDTH_M = 0.55
_MAX_WIDTH_M = 3.5


@dataclass
class Opening:
    id: int
    kind: str                     # "door" | "wide_opening"
    width_m: float
    center_px: tuple[float, float]
    rooms: list[int | str]        # two room ids, or [room_id, "exterior"]
    confidence: float


def _room_label_image(plan: PreprocessedPlan, vector: VectorPlan) -> np.ndarray:
    """Raster where pixel value = room_id + 1 (0 = not a room)."""
    img = np.zeros((plan.height, plan.width), np.int32)
    for room in vector.rooms:
        pts = np.array(room.polygon.exterior.coords, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [pts], room.id + 1)
    return img


def detect_openings(
    plan: PreprocessedPlan,
    structure: DetectedStructure,
    vector: VectorPlan,
    meters_per_px: float,
) -> list[Opening]:
    sealed = seal_walls(structure.wall_mask, structure.wall_thickness_px)
    bridge = cv2.bitwise_and(sealed, cv2.bitwise_not(structure.wall_mask))

    room_img = _room_label_image(plan, vector)
    exterior = cv2.bitwise_not(cv2.bitwise_or(sealed, (room_img > 0).astype(np.uint8) * 255))

    n, labels, cv_stats, centroids = cv2.connectedComponentsWithStats(bridge)
    openings: list[Opening] = []
    reach = max(3, int(round(structure.wall_thickness_px * 0.75)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (reach * 2 + 1, reach * 2 + 1))

    for comp in range(1, n):
        comp_mask = (labels == comp).astype(np.uint8) * 255
        near = cv2.dilate(comp_mask, kernel) > 0

        touched: set[int | str] = set(int(v) - 1 for v in np.unique(room_img[near]) if v > 0)
        if np.any(exterior[near] > 0):
            touched.add("exterior")
        if len(touched) < 2:
            continue  # a sealing artifact inside a wall junction, not an opening

        # The bridge blob spans the wall gap: its longer minAreaRect side is
        # the clear width of the opening.
        contour = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        width_m = max(w, h) * meters_per_px
        if not (_MIN_WIDTH_M <= width_m <= _MAX_WIDTH_M):
            continue

        room_ids = sorted(t for t in touched if isinstance(t, int))
        parties: list[int | str] = list(room_ids[:2])
        if len(parties) < 2 and "exterior" in touched:
            parties.append("exterior")
        openings.append(
            Opening(
                id=len(openings),
                kind="door" if width_m <= 1.6 else "wide_opening",
                width_m=round(width_m, 2),
                center_px=(round(float(centroids[comp][0]), 1), round(float(centroids[comp][1]), 1)),
                rooms=parties,
                confidence=0.7,  # geometric evidence only; no leaf/swing symbol read yet
            )
        )
    return openings
