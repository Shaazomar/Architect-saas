"""Stage 5 — Furniture AI: constraint-based placement with collision detection.

Rules, per the platform spec:
- wall-mounted pieces align to the room's walls, freestanding pieces anchor
  near the room centre;
- walking space is preserved by a per-item clearance (chairs/stools may sit
  close, large pieces keep 0.45 m circulation);
- door clearance: candidates intersecting a blocked zone (a disc around each
  detected door) are rejected;
- a piece that cannot be placed legally is skipped, never forced.

Pieces can carry an elevation (upper cabinets, mirrors), a stack of pieces
built on top of them (mattress + blanket on a bed), or be flat overlays
(rugs, prayer mats) that other furniture may stand on.

Everything works on room polygons in *meter* space (the same mirrored frame
the reconstruction stage uses).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

_CLEARANCE = 0.45   # default walking space around large pieces
_WALL_GAP = 0.05    # gap between a wall-mounted piece and the wall face

# (name, width, depth, z0, height, color)
StackPiece = tuple[str, float, float, float, float, tuple[int, int, int, int]]

_WOOD = (128, 99, 74, 255)
_LIGHT_WOOD = (152, 122, 90, 255)
_FABRIC = (121, 92, 69, 255)
_WHITE = (240, 240, 240, 255)
_STEEL = (200, 200, 205, 255)
_DARK = (62, 60, 66, 255)
_GREEN = (96, 128, 84, 255)
_GLASS = (180, 210, 220, 160)


@dataclass(frozen=True)
class CatalogItem:
    name: str
    width: float    # m, along the wall
    depth: float    # m, away from the wall
    height: float   # m
    color: tuple[int, int, int, int]
    against_wall: bool = True
    z0: float = 0.0
    overlay: bool = False       # flat floor overlay: ignores/skips collision
    clearance: float = _CLEARANCE
    count: int = 1
    stack: tuple[StackPiece, ...] = ()


_BED = CatalogItem(
    "bed", 2.0, 1.7, 0.4, _WOOD,
    stack=(
        ("mattress", 1.9, 1.6, 0.4, 0.18, _WHITE),
        ("blanket", 1.9, 1.05, 0.58, 0.05, (150, 130, 160, 255)),
    ),
)
_WC = CatalogItem("wc", 0.7, 0.75, 0.75, _WHITE)
_BASIN = CatalogItem("wash_basin", 0.6, 0.45, 0.85, _WHITE)
_MIRROR = CatalogItem("mirror", 0.6, 0.05, 0.9, _GLASS, z0=1.0)
_CHAIR = CatalogItem("chair", 0.45, 0.45, 0.9, _LIGHT_WOOD, against_wall=False, clearance=0.05)

CATALOG: dict[str, list[CatalogItem]] = {
    "living_room": [
        CatalogItem("rug", 2.6, 1.9, 0.02, (176, 154, 128, 255), against_wall=False, overlay=True),
        CatalogItem("sofa", 2.4, 0.95, 0.85, _FABRIC),
        CatalogItem("tv_unit", 1.8, 0.45, 0.55, _DARK),
        CatalogItem("coffee_table", 1.1, 0.6, 0.45, _LIGHT_WOOD, against_wall=False),
        CatalogItem("side_table", 0.5, 0.5, 0.55, _LIGHT_WOOD),
        CatalogItem("plant", 0.4, 0.4, 1.2, _GREEN),
    ],
    "dining": [
        CatalogItem(
            "dining_table", 1.8, 1.0, 0.75, _WOOD, against_wall=False,
            stack=(("centerpiece", 0.35, 0.35, 0.75, 0.3, _STEEL),),
        ),
        CatalogItem("dining_chair", 0.45, 0.45, 0.9, _LIGHT_WOOD, against_wall=False,
                    clearance=0.05, count=4),
    ],
    "kitchen": [
        CatalogItem("counter", 2.4, 0.6, 0.9, (92, 92, 96, 255),
                    stack=(("microwave", 0.5, 0.38, 0.9, 0.32, _DARK),)),
        CatalogItem("upper_cabinet", 2.4, 0.35, 0.7, _WOOD, z0=1.45),
        CatalogItem("fridge", 0.8, 0.7, 1.8, _STEEL),
        CatalogItem("island", 1.6, 0.9, 0.9, (110, 110, 116, 255), against_wall=False),
        CatalogItem("bar_stool", 0.4, 0.4, 0.65, _DARK, against_wall=False,
                    clearance=0.05, count=2),
    ],
    "bedroom": [
        _BED,
        CatalogItem("wardrobe", 1.8, 0.6, 2.1, _WOOD),
        CatalogItem("study_table", 1.2, 0.6, 0.75, _LIGHT_WOOD),
        _CHAIR,
        _MIRROR,
    ],
    "master_bedroom": [
        _BED,
        CatalogItem("wardrobe", 2.4, 0.6, 2.1, _WOOD),
        CatalogItem("study_table", 1.2, 0.6, 0.75, _LIGHT_WOOD),
        _CHAIR,
        _MIRROR,
    ],
    "bathroom": [
        _WC,
        _BASIN,
        _MIRROR,
        CatalogItem("shower_tray", 0.9, 0.9, 0.06, (198, 210, 214, 255)),
        CatalogItem("glass_partition", 0.95, 0.05, 2.0, _GLASS, clearance=0.0),
        CatalogItem("cabinet", 0.8, 0.35, 0.8, _WOOD),
    ],
    "common_toilet": [_WC, _BASIN, _MIRROR],
    "pooja_room": [
        CatalogItem("mandir", 0.9, 0.5, 1.8, _WOOD),
        CatalogItem("prayer_mat", 1.2, 0.75, 0.01, (170, 60, 50, 255),
                    against_wall=False, overlay=True),
        CatalogItem("brass_lamp", 0.25, 0.25, 0.6, (181, 141, 56, 255),
                    against_wall=False, clearance=0.2),
    ],
    "balcony": [
        CatalogItem("outdoor_table", 0.7, 0.7, 0.7, _LIGHT_WOOD, against_wall=False),
        CatalogItem("outdoor_chair", 0.5, 0.5, 0.85, _LIGHT_WOOD, against_wall=False,
                    clearance=0.05, count=2),
        CatalogItem("plant", 0.4, 0.4, 1.2, _GREEN),
    ],
    "garage": [
        CatalogItem("car", 4.2, 1.8, 1.4, (90, 100, 120, 255), against_wall=False),
        CatalogItem("storage_rack", 2.0, 0.5, 1.8, _STEEL),
        CatalogItem("tool_cabinet", 0.9, 0.45, 1.0, _DARK),
    ],
    "study": [
        CatalogItem("study_table", 1.4, 0.7, 0.75, _LIGHT_WOOD),
        _CHAIR,
        CatalogItem("bookshelf", 0.9, 0.35, 1.8, _WOOD),
    ],
    "store": [CatalogItem("storage_rack", 1.5, 0.5, 1.8, _STEEL)],
    "utility": [CatalogItem("washing_machine", 0.6, 0.6, 0.85, _WHITE)],
    "hallway": [CatalogItem("plant", 0.4, 0.4, 1.2, _GREEN)],
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
    z0: float = 0.0


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


def _find_spot(
    room: Polygon,
    inner: Polygon,
    occupied: Polygon | None,
    blocked: Polygon | None,
    item: CatalogItem,
) -> tuple[Polygon, float] | None:
    def legal(cand: Polygon) -> bool:
        if not cand.within(inner):
            return False
        if blocked is not None and cand.intersects(blocked):
            return False  # door swing / walking path
        if item.overlay or occupied is None:
            return True
        if item.z0 >= 1.2:
            return True  # elevated (upper cabinets, mirrors): clears floor pieces
        return cand.distance(occupied) >= item.clearance and not cand.intersects(occupied)

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
                    return cand, angle
    else:
        c = room.centroid
        for dx, dy in ((0, 0), (0.9, 0), (-0.9, 0), (0, 0.9), (0, -0.9),
                       (0.9, 0.9), (-0.9, -0.9), (1.6, 0), (-1.6, 0)):
            cand = _oriented_box(c.x + dx, c.y + dy, item.width, item.depth, 0)
            if legal(cand):
                return cand, 0.0
    return None


def place_furniture(
    room_id: int,
    label: str,
    room: Polygon,
    blocked: Polygon | None = None,
) -> list[PlacedItem]:
    placed: list[PlacedItem] = []
    inner = room.buffer(-_WALL_GAP)
    if inner.is_empty:
        return placed
    occupied: Polygon | None = None

    for item in CATALOG.get(label, []):
        for _ in range(item.count):
            spot = _find_spot(room, inner, occupied, blocked, item)
            if spot is None:
                break
            footprint, angle = spot
            placed.append(
                PlacedItem(item.name, footprint, item.height, item.color,
                           room_id, label, z0=item.z0)
            )
            cx, cy = footprint.centroid.coords[0]
            for s_name, s_w, s_d, s_z0, s_h, s_color in item.stack:
                placed.append(
                    PlacedItem(s_name, _oriented_box(cx, cy, s_w, s_d, angle),
                               s_h, s_color, room_id, label, z0=s_z0)
                )
            if not item.overlay:
                occupied = unary_union(
                    [p.footprint for p in placed if p.z0 < 1.2]
                )
    return placed
