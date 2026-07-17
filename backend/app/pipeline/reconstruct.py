"""Stage 6 — 2D vector plan to a 3D scene (Trimesh).

Coordinate system: X east, Y up, Z south (glTF convention). Pixel Y grows
downward in image space, so it maps directly onto +Z, which keeps the plan's
on-screen orientation when viewed from above.
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..config import settings
from .types import ReconstructionResult, VectorPlan

_LABEL_COLORS = {
    "living_room": (196, 178, 152, 255),
    "bedroom": (176, 190, 205, 255),
    "bathroom": (168, 205, 200, 255),
    "hallway": (205, 198, 176, 255),
    "room": (190, 190, 190, 255),
}
_WALL_COLOR = (232, 230, 226, 255)


def _extrude(poly: Polygon | MultiPolygon, height: float) -> list[trimesh.Trimesh]:
    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    meshes = []
    for p in polys:
        if p.is_empty or p.area <= 0:
            continue
        meshes.append(trimesh.creation.extrude_polygon(p, height))
    return meshes


def _to_gltf_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # extrude_polygon builds in XY with +Z up; -90 deg about X puts extrusion
    # height on +Y, and (with the Y mirror applied in to_m) the image's
    # downward axis on +Z.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    return mesh


def reconstruct(vector: VectorPlan, meters_per_px: float | None = None) -> ReconstructionResult:
    scale = meters_per_px or (settings.wall_thickness_m / vector.wall_thickness_px)

    def to_m(geom):
        import shapely.affinity as aff
        # Negative yfact: image Y grows downward, math Y grows upward.
        return aff.scale(geom, xfact=scale, yfact=-scale, origin=(0, 0))

    scene = trimesh.Scene()

    walls_m = to_m(vector.walls)
    wall_meshes = _extrude(walls_m, settings.wall_height_m)
    for i, m in enumerate(wall_meshes):
        m.visual.face_colors = _WALL_COLOR
        scene.add_geometry(_to_gltf_frame(m), node_name=f"wall_{i}", geom_name=f"wall_{i}")

    # One slab under the whole footprint (walls + rooms), slightly below z=0.
    footprint = unary_union([walls_m] + [to_m(r.polygon) for r in vector.rooms]).buffer(scale * 2)
    for i, m in enumerate(_extrude(footprint, settings.slab_thickness_m)):
        m.apply_translation((0, 0, -settings.slab_thickness_m))
        m.visual.face_colors = (210, 205, 196, 255)
        scene.add_geometry(_to_gltf_frame(m), node_name=f"slab_{i}", geom_name=f"slab_{i}")

    # Thin, coloured floor overlay per room so labels are visible in viewers.
    for room in vector.rooms:
        for m in _extrude(to_m(room.polygon), 0.02):
            m.visual.face_colors = _LABEL_COLORS.get(room.label, _LABEL_COLORS["room"])
            name = f"room_{room.id}_{room.label}"
            scene.add_geometry(_to_gltf_frame(m), node_name=name, geom_name=name)

    glb = scene.export(file_type="glb")
    if isinstance(glb, str):
        glb = glb.encode()

    bounds = scene.bounds
    stats = {
        "meters_per_px": scale,
        "wall_mesh_count": len(wall_meshes),
        "room_count": len(vector.rooms),
        "bounds_m": bounds.tolist() if bounds is not None else None,
        "wall_volume_m3": float(sum(m.volume for m in wall_meshes)),
    }
    return ReconstructionResult(scene_glb=glb, meters_per_px=scale, stats=stats)
