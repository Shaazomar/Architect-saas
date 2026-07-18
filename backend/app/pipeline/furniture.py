"""Furniture AI — constraint-based placement with collision detection.

Rules, per the platform spec:
- furniture is never placed randomly: wall-mounted pieces align to the room's
  walls, freestanding pieces anchor near the room centre;
- circulation is preserved by a clearance buffer around every placed piece;
- candidates that collide with other furniture or poke outside the room are
  rejected outright — a piece that cannot be placed legally is skipped.

Everything works on room polygons in *meter* space (the same mirrored frame
the reconstruction stage uses), so placements land exactly where the meshes
are extruded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

_CLEARANCE = 0.45   # m of walking space kept around each piece
_WALL_GAP = 0.05    # m between a wall-mounted piece and the wall face


@dataclass(frozen=True)
class CatalogItem:
    name: str
    width: float    # m, along the wall
    depth: float    # m, away from the wall
    height: float   # m
    color: tuple[int, int, int, int]
    against_wall: bool = True


CATALOG: dict[str, list[CatalogItem]] = {
    "living_room": [
        CatalogItem("sofa", 2.2, 0.95, 0.85, (121, 92, 69, 255)),
        CatalogItem("tv_unit", 1.8, 0.45, 0.55, (62, 60, 66, 255)),
        CatalogItem("bookshelf", 0.9, 0.35, 1.8, (140, 110, 80, 255)),
        CatalogItem("coffee_table", 1.1, 0.6, 0.45, (152, 122, 90, 255), against_wall=False),
    ],
    "bedroom": [
        CatalogItem("bed", 2.0, 1.7, 0.55, (173, 152, 173, 255)),
        CatalogItem("wardrobe", 1.8, 0.6, 2.1, (128, 99, 74, 255)),
        CatalogItem("desk", 1.2, 0.6, 0.75, (152, 122, 90, 255)),
    ],
    "master_bedroom": [
        CatalogItem("bed", 2.0, 1.8, 0.55, (173, 152, 173, 255)),
        CatalogItem("wardrobe", 2.4, 0.6, 2.1, (128, 99, 74, 255)),
        CatalogItem("desk", 1.2, 0.6, 0.75, (152, 122, 90, 255)),
    ],
    "bathroom": [
        CatalogItem("toilet", 0.7, 0.75, 0.75, (240, 240, 240, 255)),
        CatalogItem("basin", 0.6, 0.45, 0.85, (235, 235, 235, 255)),
        CatalogItem("shower_tray", 0.9, 0.9, 0.06, (198, 210, 214, 255)),
    ],
    "kitchen": [
        CatalogItem("counter", 2.4, 0.6, 0.9, (92, 92, 96, 255)),
        CatalogItem("fridge", 0.8, 0.7, 1.8, (200, 200, 205, 255)),
        CatalogItem("island", 1.6, 0.9, 0.9, (110, 110, 116, 255), against_wall=False),
    ],
    "hallway": [],
    "room": [],
}


@dataclass
class PlacedItem:
    name: str
    footprint: Polygon   # meter space
    height: float
    color: tuple[int, int, int, int]
    room_id: int
    room_label: str


def _edges_longest_first(room: Polygon) -> list[tuple[tuple, tuple, float]]:
    coords = list(room.exterior.coords)
    edges = []
    for p1, p2 in zip(coords, coords[1:]):
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length > 0.6:
            edges.append((p1, p2, length))
    edges.sort(key=lambda e: -e[2])
    return edges


def _oriented_box(cx: float, cy: float, w: float, d: float, angle_deg: float) -> Polygon:
    b = box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2)
    return affinity.rotate(b, angle_deg, origin=(cx, cy))


def _find_spot(room: Polygon, inner: Polygon, occupied: Polygon | None, item: CatalogItem) -> Polygon | None:
    def legal(cand: Polygon) -> bool:
        return cand.within(inner) and (occupied is None or not cand.intersects(occupied))

    if item.against_wall:
        for p1, p2, length in _edges_longest_first(room):
            if length < item.width + 0.2:
                continue
            ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
            nx, ny = -uy, ux
            mid = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            if not room.contains(Point(mid[0] + nx * 0.1, mid[1] + ny * 0.1)):
                nx, ny = -nx, -ny  # flip to the inward normal
            angle = math.degrees(math.atan2(uy, ux))
            offset = item.depth / 2 + _WALL_GAP
            for t in (0.5, 0.3, 0.7, 0.2, 0.8):
                cx = p1[0] + ux * length * t + nx * offset
                cy = p1[1] + uy * length * t + ny * offset
                cand = _oriented_box(cx, cy, item.width, item.depth, angle)
                if legal(cand):
                    return cand
    else:
        c = room.centroid
        for dx, dy in ((0, 0), (0.8, 0), (-0.8, 0), (0, 0.8), (0, -0.8)):
            cand = _oriented_box(c.x + dx, c.y + dy, item.width, item.depth, 0)
            if legal(cand):
                return cand
    return None


def place_furniture(room_id: int, label: str, room: Polygon) -> list[PlacedItem]:
    placed: list[PlacedItem] = []
    inner = room.buffer(-_WALL_GAP)
    if inner.is_empty:
        return placed
    occupied: Polygon | None = None
    for item in CATALOG.get(label, []):
        spot = _find_spot(room, inner, occupied, item)
        if spot is None:
            continue
        placed.append(PlacedItem(item.name, spot, item.height, item.color, room_id, label))
        # Clearance buffer preserves circulation between pieces.
        occupied = unary_union([p.footprint for p in placed]).buffer(_CLEARANCE)
    return placed
