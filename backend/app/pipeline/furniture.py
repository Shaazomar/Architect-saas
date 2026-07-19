"""Stages 5 + 7 — Furniture & Decor AI: constraint-based placement.

Rules:
- wall-mounted pieces align to the room's walls, freestanding pieces anchor
  near the room centre;
- walking space is preserved by a per-item clearance (chairs/stools may sit
  close, large pieces keep 0.45 m circulation);
- door clearance: candidates intersecting a blocked zone (a disc around each
  detected door) are rejected;
- a piece that cannot be placed legally is skipped, never forced;
- decor is bounded per room — never overdecorate.

Pieces can carry an elevation (upper cabinets, mirrors, wall art), a stack of
pieces built on/around them with local offsets (mattress, pillows at the
head, blanket at the foot; appliances on counters), or be flat overlays
(rugs, prayer mats) that other furniture may stand on.

Everything works on room polygons in *meter* space (the same mirrored frame
the reconstruction stage uses). Materials are resolved from item names by
the Stage 6 material system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

_CLEARANCE = 0.45   # default walking space around large pieces
_WALL_GAP = 0.05    # gap between a wall-mounted piece and the wall face

# (name, width, depth, dx, dy_wall, z0, height)
# dx: local offset along the item's width axis; dy_wall: offset toward (+) or
# away from (-) the wall the base piece stands against.
StackPiece = tuple[str, float, float, float, float, float, float]


@dataclass(frozen=True)
class CatalogItem:
    name: str
    width: float    # m, along the wall
    depth: float    # m, away from the wall
    height: float   # m
    against_wall: bool = True
    z0: float = 0.0
    overlay: bool = False       # flat floor overlay: ignores/skips collision
    clearance: float = _CLEARANCE
    count: int = 1
    decor: bool = False         # Stage 7 decoration, not primary furniture
    stack: tuple[StackPiece, ...] = ()


_BED = CatalogItem(
    "bed", 2.0, 1.7, 0.4,
    stack=(
        ("mattress", 1.9, 1.6, 0.0, 0.0, 0.4, 0.18),
        ("blanket", 1.9, 1.0, 0.0, -0.3, 0.58, 0.05),
        ("pillow_left", 0.55, 0.35, -0.45, 0.55, 0.58, 0.12),
        ("pillow_right", 0.55, 0.35, 0.45, 0.55, 0.58, 0.12),
    ),
)
_WC = CatalogItem("wc", 0.7, 0.75, 0.75)
_BASIN = CatalogItem("wash_basin", 0.6, 0.45, 0.85)
_MIRROR = CatalogItem("mirror", 0.6, 0.05, 0.9, z0=1.0)
_CHAIR = CatalogItem("chair", 0.45, 0.45, 0.9, against_wall=False, clearance=0.05)
_PLANT = CatalogItem("plant", 0.4, 0.4, 1.2, decor=True)

CATALOG: dict[str, list[CatalogItem]] = {
    "living_room": [
        CatalogItem("rug", 2.6, 1.9, 0.02, against_wall=False, overlay=True),
        CatalogItem("sofa", 2.4, 0.95, 0.85),
        CatalogItem("tv_unit", 1.8, 0.45, 0.55),
        CatalogItem("coffee_table", 1.1, 0.6, 0.45, against_wall=False,
                    stack=(("books", 0.35, 0.25, 0.25, 0.0, 0.45, 0.12),)),
        CatalogItem("side_table", 0.5, 0.5, 0.55),
        # --- decor (Stage 7): bounded, realistic ---
        CatalogItem("wall_art", 1.2, 0.04, 0.8, z0=1.3, decor=True),
        CatalogItem("floor_lamp", 0.35, 0.35, 1.6, decor=True),
        _PLANT,
        CatalogItem("led_strip", 2.0, 0.04, 0.03, z0=2.8, clearance=0.0, decor=True),
    ],
    "dining": [
        CatalogItem(
            "dining_table", 1.8, 1.0, 0.75, against_wall=False,
            stack=(("centerpiece", 0.35, 0.35, 0.0, 0.0, 0.75, 0.3),),
        ),
        CatalogItem("dining_chair", 0.45, 0.45, 0.9, against_wall=False,
                    clearance=0.05, count=4),
    ],
    "kitchen": [
        CatalogItem(
            "counter", 2.4, 0.6, 0.9,
            stack=(
                ("microwave", 0.5, 0.38, -0.7, 0.05, 0.9, 0.32),
                ("coffee_machine", 0.3, 0.35, 0.75, 0.05, 0.9, 0.35),
                ("knife_stand", 0.2, 0.15, 0.25, 0.1, 0.9, 0.25),
            ),
        ),
        CatalogItem("upper_cabinet", 2.4, 0.35, 0.7, z0=1.45),
        CatalogItem("fridge", 0.8, 0.7, 1.8),
        CatalogItem("island", 1.6, 0.9, 0.9, against_wall=False,
                    stack=(("fruit_bowl", 0.3, 0.3, 0.0, 0.0, 0.9, 0.15),)),
        CatalogItem("bar_stool", 0.4, 0.4, 0.65, against_wall=False,
                    clearance=0.05, count=2),
    ],
    "bedroom": [
        _BED,
        CatalogItem("wardrobe", 1.8, 0.6, 2.1),
        CatalogItem("study_table", 1.2, 0.6, 0.75),
        _CHAIR,
        _MIRROR,
        _PLANT,
    ],
    "master_bedroom": [
        _BED,
        CatalogItem("wardrobe", 2.4, 0.6, 2.1),
        CatalogItem("study_table", 1.2, 0.6, 0.75),
        _CHAIR,
        _MIRROR,
        _PLANT,
    ],
    "bathroom": [
        _WC,
        _BASIN,
        _MIRROR,
        CatalogItem("shower_tray", 0.9, 0.9, 0.06),
        CatalogItem("glass_partition", 0.95, 0.05, 2.0, clearance=0.0),
        CatalogItem("cabinet", 0.8, 0.35, 0.8),
        CatalogItem("towel", 0.6, 0.06, 0.5, z0=0.9, decor=True),
    ],
    "common_toilet": [_WC, _BASIN, _MIRROR,
                      CatalogItem("towel", 0.6, 0.06, 0.5, z0=0.9, decor=True)],
    "pooja_room": [
        CatalogItem("mandir", 0.9, 0.5, 1.8),
        CatalogItem("prayer_mat", 1.2, 0.75, 0.01, against_wall=False, overlay=True),
        CatalogItem("brass_lamp", 0.25, 0.25, 0.6, against_wall=False,
                    clearance=0.2, decor=True),
    ],
    "balcony": [
        CatalogItem("outdoor_table", 0.7, 0.7, 0.7, against_wall=False),
        CatalogItem("outdoor_chair", 0.5, 0.5, 0.85, against_wall=False,
                    clearance=0.05, count=2),
        _PLANT,
        CatalogItem("led_strip", 1.5, 0.04, 0.03, z0=2.6, clearance=0.0, decor=True),
    ],
    "garage": [
        CatalogItem("car", 4.2, 1.8, 1.4, against_wall=False),
        CatalogItem("storage_rack", 2.0, 0.5, 1.8),
        CatalogItem("tool_cabinet", 0.9, 0.45, 1.0),
    ],
    "study": [
        CatalogItem("study_table", 1.4, 0.7, 0.75),
        _CHAIR,
        CatalogItem("bookshelf", 0.9, 0.35, 1.8),
        _PLANT,
    ],
    "store": [CatalogItem("storage_rack", 1.5, 0.5, 1.8)],
    "utility": [CatalogItem("washing_machine", 0.6, 0.6, 0.85)],
    "hallway": [_PLANT],
    "room": [],
}


@dataclass
class PlacedItem:
    name: str
    footprint: Polygon   # meter space
    height: float
    room_id: int
    room_label: str
    z0: float = 0.0
    decor: bool = False


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
) -> tuple[Polygon, float, float] | None:
    """Returns (footprint, angle_deg, wall_sign): wall_sign maps local +y to
    toward-the-wall (+1) or away (-1); freestanding pieces get -1."""

    def legal(cand: Polygon) -> bool:
        if not cand.within(inner):
            return False
        if blocked is not None and cand.intersects(blocked):
            return False  # door swing / walking path
        if item.overlay or occupied is None:
            return True
        if item.z0 >= 1.2:
            return True  # elevated (upper cabinets, mirrors, art): clears floor pieces
        return cand.distance(occupied) >= item.clearance and not cand.intersects(occupied)

    if item.against_wall:
        for p1, p2, length in _edges_longest_first(room):
            if length < item.width + 0.2:
                continue
            ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
            nx, ny = -uy, ux
            flipped = False
            mid = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            if not room.contains(Point(mid[0] + nx * 0.1, mid[1] + ny * 0.1)):
                nx, ny, flipped = -nx, -ny, True  # flip to the inward normal
            angle = math.degrees(math.atan2(uy, ux))
            # Local +y (after rotation) equals the *unflipped* normal, so it
            # points toward the wall exactly when the normal was flipped.
            wall_sign = 1.0 if flipped else -1.0
            offset = item.depth / 2 + _WALL_GAP
            for t in (0.5, 0.3, 0.7, 0.2, 0.8):
                cx = p1[0] + ux * length * t + nx * offset
                cy = p1[1] + uy * length * t + ny * offset
                cand = _oriented_box(cx, cy, item.width, item.depth, angle)
                if legal(cand):
                    return cand, angle, wall_sign
    else:
        c = room.centroid
        for dx, dy in ((0, 0), (0.9, 0), (-0.9, 0), (0, 0.9), (0, -0.9),
                       (0.9, 0.9), (-0.9, -0.9), (1.6, 0), (-1.6, 0)):
            cand = _oriented_box(c.x + dx, c.y + dy, item.width, item.depth, 0)
            if legal(cand):
                return cand, 0.0, -1.0
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
            footprint, angle, wall_sign = spot
            placed.append(
                PlacedItem(item.name, footprint, item.height,
                           room_id, label, z0=item.z0, decor=item.decor)
            )
            cx, cy = footprint.centroid.coords[0]
            rad = math.radians(angle)
            for s_name, s_w, s_d, s_dx, s_dy_wall, s_z0, s_h in item.stack:
                dy = s_dy_wall * wall_sign
                ox = cx + s_dx * math.cos(rad) - dy * math.sin(rad)
                oy = cy + s_dx * math.sin(rad) + dy * math.cos(rad)
                placed.append(
                    PlacedItem(s_name, _oriented_box(ox, oy, s_w, s_d, angle),
                               s_h, room_id, label, z0=s_z0, decor=item.decor)
                )
            if not item.overlay:
                occupied = unary_union(
                    [p.footprint for p in placed if p.z0 < 1.2]
                )
    return placed
