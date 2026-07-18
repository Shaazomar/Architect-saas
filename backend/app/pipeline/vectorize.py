"""Stage 3 — raster masks to clean vector geometry (Shapely, pixel coords)."""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .preprocess import PlanImageError
from .types import DetectedStructure, PreprocessedPlan, Room, VectorPlan

_MIN_ROOM_AREA_FACTOR = 12.0   # a room must exceed (wall_thickness^2 * factor)
_SIMPLIFY_TOLERANCE = 1.5      # px; keeps corners crisp, drops raster jaggies


def _mask_to_polygons(mask: np.ndarray, min_area: float) -> list[Polygon]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    polys: list[Polygon] = []
    hierarchy = hierarchy[0]
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1 or len(contour) < 3:  # only outer contours here
            continue
        shell = contour[:, 0, :].astype(float)
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            hole = contours[child]
            if len(hole) >= 3:
                holes.append(hole[:, 0, :].astype(float))
            child = hierarchy[child][0]
        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area:
            continue
        poly = poly.simplify(_SIMPLIFY_TOLERANCE, preserve_topology=True)
        if not poly.is_empty:
            polys.append(poly)
    return polys


def seal_walls(walls: np.ndarray, thickness: float) -> np.ndarray:
    """Close door/passage gaps in the wall mask. Door openings are ~4x wall
    thickness in practice; 7x with margin makes the closing bridge every
    opening so rooms separate cleanly. Also used by the openings detector:
    sealed minus original walls = exactly the openings."""
    seal = max(3, int(round(thickness * 7)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (seal, seal))
    return cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)


def _room_free_space(plan: PreprocessedPlan, walls: np.ndarray, thickness: float) -> np.ndarray:
    """Interior free space: seal door openings, then flood-fill away the
    exterior so only enclosed rooms remain."""
    free = cv2.bitwise_not(seal_walls(walls, thickness))
    # Flood fill from every border pixel to remove the outside world.
    ff_mask = np.zeros((plan.height + 2, plan.width + 2), np.uint8)
    flooded = free.copy()
    for x in (0, plan.width - 1):
        for y in range(0, plan.height, 16):
            if flooded[y, x] == 255:
                cv2.floodFill(flooded, ff_mask, (x, y), 0)
    for y in (0, plan.height - 1):
        for x in range(0, plan.width, 16):
            if flooded[y, x] == 255:
                cv2.floodFill(flooded, ff_mask, (x, y), 0)
    return flooded


def vectorize(plan: PreprocessedPlan, structure: DetectedStructure) -> VectorPlan:
    t = structure.wall_thickness_px

    wall_polys = _mask_to_polygons(structure.wall_mask, min_area=t * t * 2)
    if not wall_polys:
        raise PlanImageError("Wall geometry could not be vectorized.")
    walls = unary_union(wall_polys)
    if isinstance(walls, Polygon):
        walls = MultiPolygon([walls])

    interior = _room_free_space(plan, structure.wall_mask, t)
    n, labels = cv2.connectedComponents(interior)
    rooms: list[Room] = []
    min_room_area = t * t * _MIN_ROOM_AREA_FACTOR
    for comp in range(1, n):
        comp_mask = np.where(labels == comp, np.uint8(255), np.uint8(0))
        for poly in _mask_to_polygons(comp_mask, min_area=min_room_area):
            rooms.append(Room(id=len(rooms), polygon=poly, area_px=poly.area))

    if not rooms:
        raise PlanImageError("No enclosed rooms were found in the drawing.")

    return VectorPlan(walls=walls, rooms=rooms, wall_thickness_px=t)
