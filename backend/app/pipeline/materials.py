"""Stage 6 — Material AI: PBR (metallic-roughness) material assignment.

Every mesh gets a real glTF PBR material — base color, metallic and
roughness factors (plus emissive for light strips), chosen from what the
element *is*: walls get paint, wet-area floors get ceramic, bedroom floors
get wood, counters get granite, mirrors get polished metal, glass gets
alpha-blended transmission approximation.

Texture maps (normal / AO / UV-mapped albedo) belong to the texture
pipeline stage and are reported as pending — factors are never faked into
pretend texture maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import trimesh


@dataclass(frozen=True)
class Material:
    name: str
    base_color: tuple[int, int, int, int]
    metallic: float
    roughness: float
    emissive: tuple[float, float, float] | None = None
    alpha_blend: bool = False


_M = Material
MATERIALS: dict[str, Material] = {
    "wall_paint": _M("wall_paint", (235, 232, 226, 255), 0.0, 0.85),
    "concrete": _M("concrete", (205, 200, 192, 255), 0.0, 0.95),
    "marble_floor": _M("marble_floor", (222, 218, 210, 255), 0.05, 0.25),
    "wood_floor": _M("wood_floor", (188, 154, 116, 255), 0.0, 0.6),
    "ceramic_tile": _M("ceramic_tile", (214, 220, 222, 255), 0.0, 0.3),
    "wood_dark": _M("wood_dark", (110, 84, 60, 255), 0.0, 0.65),
    "wood": _M("wood", (128, 99, 74, 255), 0.0, 0.65),
    "wood_light": _M("wood_light", (152, 122, 90, 255), 0.0, 0.6),
    "fabric": _M("fabric", (121, 92, 69, 255), 0.0, 0.95),
    "fabric_light": _M("fabric_light", (222, 218, 214, 255), 0.0, 0.95),
    "fabric_accent": _M("fabric_accent", (150, 130, 160, 255), 0.0, 0.95),
    "granite": _M("granite", (70, 70, 76, 255), 0.05, 0.35),
    "steel": _M("steel", (200, 200, 205, 255), 0.9, 0.35),
    "ceramic_white": _M("ceramic_white", (245, 245, 245, 255), 0.0, 0.25),
    "mirror": _M("mirror", (230, 235, 240, 255), 1.0, 0.05),
    "glass": _M("glass", (185, 215, 225, 110), 0.0, 0.08, alpha_blend=True),
    "brass": _M("brass", (181, 141, 56, 255), 1.0, 0.4),
    "plant": _M("plant", (96, 128, 84, 255), 0.0, 0.9),
    "car_paint": _M("car_paint", (90, 100, 120, 255), 0.6, 0.35),
    "rug": _M("rug", (176, 154, 128, 255), 0.0, 1.0),
    "led_strip": _M("led_strip", (255, 250, 235, 255), 0.0, 0.4, emissive=(1.0, 0.95, 0.85)),
    "dark_panel": _M("dark_panel", (62, 60, 66, 255), 0.1, 0.5),
    "art": _M("art", (170, 120, 90, 255), 0.0, 0.7),
}

_FLOOR_BY_ROOM = {
    "living_room": "marble_floor",
    "hallway": "marble_floor",
    "dining": "marble_floor",
    "kitchen": "ceramic_tile",
    "bathroom": "ceramic_tile",
    "common_toilet": "ceramic_tile",
    "utility": "ceramic_tile",
    "bedroom": "wood_floor",
    "master_bedroom": "wood_floor",
    "study": "wood_floor",
    "pooja_room": "marble_floor",
    "balcony": "ceramic_tile",
    "garage": "concrete",
    "store": "concrete",
}

_ITEM_MATERIALS = {
    "sofa": "fabric", "rug": "rug", "prayer_mat": "fabric_accent",
    "mattress": "fabric_light", "pillow": "fabric_light", "blanket": "fabric_accent",
    "curtain": "fabric_accent", "towel": "fabric_light",
    "wc": "ceramic_white", "wash_basin": "ceramic_white", "shower_tray": "ceramic_tile",
    "mirror": "mirror", "glass_partition": "glass",
    "fridge": "steel", "microwave": "dark_panel", "washing_machine": "ceramic_white",
    "storage_rack": "steel", "knife_stand": "steel", "coffee_machine": "dark_panel",
    "counter": "granite", "island": "granite",
    "upper_cabinet": "wood", "cabinet": "wood", "wardrobe": "wood", "mandir": "wood",
    "tv_unit": "dark_panel", "bar_stool": "dark_panel",
    "brass_lamp": "brass", "bell": "brass", "fruit_bowl": "ceramic_white",
    "plant": "plant", "car": "car_paint",
    "wall_art": "art", "books": "fabric_accent", "floor_lamp": "brass",
    "led_strip": "led_strip", "centerpiece": "steel",
}


def material_for(node_name: str, room_label: str | None = None, item: str | None = None) -> Material:
    """Pick the material for a scene node from what the element is."""
    if item is not None:
        for key, mat in _ITEM_MATERIALS.items():
            if item.startswith(key):
                return MATERIALS[mat]
        return MATERIALS["wood_light"]

    prefix = node_name.split("_")[0]
    if prefix == "wall" or node_name.startswith("lintel") or node_name.startswith("roof_parapet"):
        return MATERIALS["wall_paint"]
    if prefix in ("slab", "roof"):
        return MATERIALS["concrete"]
    if prefix == "skirting":
        return MATERIALS["wood_dark"]
    if node_name.startswith("door_frame"):
        return MATERIALS["wood_dark"]
    if prefix == "room":
        return MATERIALS[_FLOOR_BY_ROOM.get(room_label or "", "ceramic_tile")]
    return MATERIALS["wood_light"]


def apply_material(mesh: trimesh.Trimesh, mat: Material) -> None:
    c = mat.base_color
    pbr = trimesh.visual.material.PBRMaterial(
        name=mat.name,
        baseColorFactor=[c[0], c[1], c[2], c[3]],
        metallicFactor=mat.metallic,
        roughnessFactor=mat.roughness,
    )
    if mat.emissive is not None:
        pbr.emissiveFactor = list(mat.emissive)
    if mat.alpha_blend:
        pbr.alphaMode = "BLEND"
    mesh.visual = trimesh.visual.TextureVisuals(material=pbr)
