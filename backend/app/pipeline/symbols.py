"""Furniture symbol detection — objects come from the drawing, not a catalog.

Per the fidelity mandate: never invent, move, rotate, or replace furniture.
A detected symbol is reconstructed as a BIM object with its *exact* drawn
footprint and pose. Classification is best-effort (geometry + room context)
and carries an explicit confidence; when the evidence is weak the category
stays the honest "furniture".

Detection: furniture symbols are the thin-stroke closed shapes that survive
inside rooms once walls are removed — text, dimension lines and hatching are
rejected by size, aspect and convexity filters, and by requiring the shape to
sit inside a detected room.

This module is the classical-CV stand-in for the future trained symbol
detector; it honours the same DetectedObject contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon

from .types import DetectedStructure, PreprocessedPlan, VectorPlan

_MIN_AREA_M2 = 0.15
_MAX_AREA_M2 = 10.0
_MAX_ASPECT = 5.0
_MIN_SIDE_M = 0.35      # nothing on a plan narrower than this is furniture
_MIN_CONVEXITY = 0.75   # contour area / convex hull area
_MAX_INK_RATIO = 0.4    # furniture symbols are outlines (hollow); text blobs
                        # are dense with strokes — this is what separates them
_MIN_IN_ROOM = 0.7      # fraction of the footprint inside its room

_HEIGHTS_M = {
    "bed": 0.55,
    "table": 0.75,
    "round_table": 0.75,
    "sofa": 0.85,
    "fixture": 0.45,
    "furniture": 0.75,
}
_COLOR = (150, 128, 108, 255)


@dataclass
class DetectedObject:
    id: int
    category: str
    confidence: float
    room_id: int
    room_label: str
    footprint_px: Polygon      # exact drawn footprint, pixel coordinates
    size_m: tuple[float, float]
    rotation_deg: float
    height_m: float


def _classify(w_m: float, h_m: float, circularity: float, room_label: str) -> tuple[str, float]:
    area = w_m * h_m
    if circularity > 0.82:
        return "round_table", 0.6
    if room_label in ("bedroom", "master_bedroom") and area >= 2.4:
        return "bed", 0.55
    if room_label in ("bathroom", "common_toilet") and area <= 1.5:
        return "fixture", 0.5
    if room_label == "living_room" and max(w_m, h_m) / max(0.01, min(w_m, h_m)) >= 2.2:
        return "sofa", 0.45
    if 0.3 <= area <= 4.0:
        return "table", 0.4
    return "furniture", 0.3


def detect_symbols(
    plan: PreprocessedPlan,
    structure: DetectedStructure,
    vector: VectorPlan,
    meters_per_px: float,
) -> list[DetectedObject]:
    # Everything drawn that is not wall material.
    wall_zone = cv2.dilate(
        structure.wall_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    thin = cv2.bitwise_and(plan.ink, cv2.bitwise_not(wall_zone))
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[DetectedObject] = []

    for contour in contours:
        if len(contour) < 5:
            continue
        area_px = cv2.contourArea(contour)
        area_m2 = area_px * meters_per_px**2
        if not (_MIN_AREA_M2 <= area_m2 <= _MAX_AREA_M2):
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area_px / hull_area < _MIN_CONVEXITY:
            continue  # text blocks / hatching are far from convex

        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) <= 0 or max(w, h) / min(w, h) > _MAX_ASPECT:
            continue  # dimension lines and long annotation strokes
        if min(w, h) * meters_per_px < _MIN_SIDE_M:
            continue  # too narrow to be furniture (text lines, ticks)

        # Outline-vs-text: measure how much ink fills the contour's interior.
        x, y, bw, bh = cv2.boundingRect(contour)
        region = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(region, [contour - [x, y]], -1, 255, cv2.FILLED)
        ink_inside = cv2.countNonZero(cv2.bitwise_and(plan.ink[y : y + bh, x : x + bw], region))
        if ink_inside / max(1.0, area_px) > _MAX_INK_RATIO:
            continue  # dense stroke blob: merged text, hatching — not a symbol

        poly = Polygon(contour[:, 0, :].astype(float)).buffer(0)
        if poly.is_empty:
            continue
        poly = poly.simplify(1.5, preserve_topology=True)

        # Exact-position rule: the object must belong to a detected room.
        host = None
        for room in vector.rooms:
            inter = poly.intersection(room.polygon).area
            if poly.area > 0 and inter / poly.area >= _MIN_IN_ROOM:
                host = room
                break
        if host is None:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area_px / (perimeter**2) if perimeter > 0 else 0.0
        w_m, h_m = w * meters_per_px, h * meters_per_px
        category, conf = _classify(w_m, h_m, circularity, host.label)

        objects.append(
            DetectedObject(
                id=len(objects),
                category=category,
                confidence=conf,
                room_id=host.id,
                room_label=host.label,
                footprint_px=poly,
                size_m=(round(w_m, 2), round(h_m, 2)),
                rotation_deg=round(float(angle), 1),
                height_m=_HEIGHTS_M.get(category, _HEIGHTS_M["furniture"]),
            )
        )
    return objects
