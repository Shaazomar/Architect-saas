"""Stage 4 — Geometry Engine: 2D vector plan to a 3D scene (Trimesh).

Coordinate system: X east, Y up, Z south (glTF convention). Pixel Y grows
downward in image space; to_m mirrors it so the plan keeps its drawn
orientation when viewed from above.

Generated element classes: walls, floor slab, roof/ceiling slab, parapet,
door lintels + frames (at *detected* openings only), skirting, per-room
floor finishes, and furniture (Stage 5). Window frames and glass are not
generated until the window detector lands — no windows are invented.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ..config import settings
from .furniture import _oriented_box, place_furniture
from .materials import MATERIALS, Material, apply_material, material_for
from .types import ReconstructionResult, VectorPlan

_DOOR_HEAD_M = 2.1     # standard door head height
_PARAPET_H_M = 0.9
_PARAPET_T_M = 0.15
_SKIRTING_H_M = 0.1
_SKIRTING_T_M = 0.02
_DOOR_CLEAR_M = 0.6    # walking clearance kept beyond each door's half-width

FURNITURE_MODES = ("auto", "detected", "generated", "none")


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


def reconstruct(
    vector: VectorPlan,
    meters_per_px: float | None = None,
    furniture_mode: str = "auto",
    detected_objects: list | None = None,
    openings: list | None = None,
    include_roof: bool = True,
) -> ReconstructionResult:
    """furniture_mode:
    "auto"      (default) rooms keep their drawn symbols at exact position;
                rooms with no symbols are furnished from the catalog so no
                room is left empty;
    "detected"  strictly what the drawing shows;
    "generated" catalog placement everywhere;
    "none"      structure only."""
    scale = meters_per_px or (settings.wall_thickness_m / vector.wall_thickness_px)
    openings = openings or []

    def to_m(geom):
        import shapely.affinity as aff
        # Negative yfact: image Y grows downward, math Y grows upward.
        return aff.scale(geom, xfact=scale, yfact=-scale, origin=(0, 0))

    material_assignments: dict[str, str] = {}

    def add(mesh: trimesh.Trimesh, name: str, mat: Material, z0: float = 0.0) -> None:
        if z0:
            mesh.apply_translation((0, 0, z0))
        apply_material(mesh, mat)
        material_assignments[name] = mat.name
        scene.add_geometry(_to_gltf_frame(mesh), node_name=name, geom_name=name)

    scene = trimesh.Scene()

    walls_m = to_m(vector.walls)
    wall_meshes = _extrude(walls_m, settings.wall_height_m)
    for i, m in enumerate(wall_meshes):
        add(m, f"wall_{i}", MATERIALS["wall_paint"])

    # One slab under the whole footprint (walls + rooms), slightly below z=0.
    footprint = unary_union([walls_m] + [to_m(r.polygon) for r in vector.rooms]).buffer(scale * 2)
    for i, m in enumerate(_extrude(footprint, settings.slab_thickness_m)):
        add(m, f"slab_{i}", MATERIALS["concrete"], z0=-settings.slab_thickness_m)

    # Per-room floor finish + skirting along the room boundary.
    for room in vector.rooms:
        room_m = to_m(room.polygon)
        name = f"room_{room.id}_{room.label}"
        for m in _extrude(room_m, 0.02):
            add(m, name, material_for(name, room_label=room.label))
        ring = room_m.difference(room_m.buffer(-_SKIRTING_T_M))
        for j, m in enumerate(_extrude(ring, _SKIRTING_H_M)):
            add(m, f"skirting_{room.id}_{j}", MATERIALS["wood_dark"])

    # Door lintels + frames at each *detected* opening: wall material above
    # the door head, and jamb posts on either side of the clear width.
    for o in openings:
        cx, cy = o.center_px[0] * scale, -o.center_px[1] * scale
        angle = -o.angle_deg  # Y mirror flips rotation direction
        width = o.width_px * scale
        depth = max(o.depth_px * scale, 0.08)
        lintel = _oriented_box(cx, cy, width, depth, angle)
        for m in _extrude(lintel, settings.wall_height_m - _DOOR_HEAD_M):
            add(m, f"lintel_{o.id}", MATERIALS["wall_paint"], z0=_DOOR_HEAD_M)
        rad = math.radians(angle)
        ux, uy = math.cos(rad), math.sin(rad)
        for side, sgn in (("a", 1.0), ("b", -1.0)):
            px = cx + sgn * ux * (width / 2 - 0.04)
            py = cy + sgn * uy * (width / 2 - 0.04)
            post = _oriented_box(px, py, 0.08, depth + 0.04, angle)
            for m in _extrude(post, _DOOR_HEAD_M):
                add(m, f"door_frame_{o.id}_{side}", MATERIALS["wood_dark"])

    # Roof/ceiling slab + parapet, separately named so viewers can toggle.
    if include_roof:
        for i, m in enumerate(_extrude(footprint, settings.roof_thickness_m)):
            add(m, f"roof_{i}", MATERIALS["concrete"], z0=settings.wall_height_m)
        parapet_ring = footprint.difference(footprint.buffer(-_PARAPET_T_M))
        for i, m in enumerate(_extrude(parapet_ring, _PARAPET_H_M)):
            add(m, f"roof_parapet_{i}",
                MATERIALS["wall_paint"], z0=settings.wall_height_m + settings.roof_thickness_m)

    # ---- Stage 5: furniture ----
    # Door clearance zones: no generated furniture inside them.
    blocked = None
    if openings:
        blocked = unary_union(
            [
                Point(o.center_px[0] * scale, -o.center_px[1] * scale).buffer(
                    o.width_m / 2 + _DOOR_CLEAR_M
                )
                for o in openings
            ]
        )

    detected_by_room: dict[int, list] = {}
    for obj in detected_objects or []:
        detected_by_room.setdefault(obj.room_id, []).append(obj)

    furniture_report: list[dict] = []
    furniture_nodes: dict[str, list[str]] = {"detected": [], "generated": []}

    def add_detected(objs) -> None:
        for obj in objs:
            fp_m = to_m(obj.footprint_px)
            name = f"furniture_{obj.room_id}_{obj.category}_{obj.id}"
            furniture_nodes["detected"].append(name)
            for m in _extrude(fp_m, obj.height_m):
                add(m, name, material_for("furniture", item=obj.category))
            cx, cy = fp_m.centroid.coords[0]
            furniture_report.append(
                {
                    "room_id": obj.room_id, "room_label": obj.room_label,
                    "item": obj.category, "source": "detected", "decor": False,
                    "confidence": obj.confidence, "size_m": list(obj.size_m),
                    "rotation_deg": obj.rotation_deg, "height_m": obj.height_m,
                    "footprint_m2": round(fp_m.area, 2),
                    "position_m": [round(cx, 2), round(cy, 2)],
                }
            )

    def add_generated(room) -> None:
        room_m = to_m(room.polygon)
        for idx, item in enumerate(place_furniture(room.id, room.label, room_m, blocked)):
            name = f"furniture_{item.room_id}_{item.name}_{idx}"
            furniture_nodes["generated"].append(name)
            for m in _extrude(item.footprint, item.height):
                add(m, name, material_for("furniture", item=item.name), z0=item.z0)
            cx, cy = item.footprint.centroid.coords[0]
            furniture_report.append(
                {
                    "room_id": item.room_id, "room_label": item.room_label,
                    "item": item.name, "source": "generated", "confidence": None,
                    "decor": item.decor,
                    "height_m": item.height, "z0_m": item.z0,
                    "footprint_m2": round(item.footprint.area, 2),
                    "position_m": [round(cx, 2), round(cy, 2)],
                }
            )

    if furniture_mode == "detected":
        add_detected(detected_objects or [])
    elif furniture_mode == "generated":
        for room in vector.rooms:
            add_generated(room)
    elif furniture_mode == "auto":
        for room in vector.rooms:
            objs = detected_by_room.get(room.id)
            if objs:
                add_detected(objs)   # fidelity: keep what the drawing shows
            else:
                add_generated(room)  # completeness: no empty rooms

    glb = scene.export(file_type="glb")
    if isinstance(glb, str):
        glb = glb.encode()

    counts: dict[str, int] = {}
    for name in scene.geometry:
        prefix = name.split("_")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    geometry_manifest = {
        "stage": "geometry",
        "format": "glTF binary (GLB) scene graph",
        "node_counts": counts,
        "standards_m": {
            "wall_height": settings.wall_height_m,
            "slab_thickness": settings.slab_thickness_m,
            "roof_thickness": settings.roof_thickness_m,
            "door_head": _DOOR_HEAD_M,
            "skirting_height": _SKIRTING_H_M,
            "parapet_height": _PARAPET_H_M,
        },
        "pending": {
            "window_frames_glass": "no windows detected; window detector pending",
            "stairs": "stair detector pending",
        },
        "furniture_nodes": furniture_nodes,
    }

    bounds = scene.bounds
    stats = {
        "meters_per_px": scale,
        "wall_mesh_count": len(wall_meshes),
        "room_count": len(vector.rooms),
        "bounds_m": bounds.tolist() if bounds is not None else None,
        "wall_volume_m3": float(sum(m.volume for m in wall_meshes)),
        "footprint_area_m2": float(footprint.area),
        "furniture_count": len(furniture_report),
    }
    materials_manifest = {
        "stage": "materials",
        "workflow": "pbr_metallic_roughness",
        "materials": {
            name: {
                "base_color": list(mat.base_color),
                "metallic": mat.metallic,
                "roughness": mat.roughness,
                "emissive": list(mat.emissive) if mat.emissive else None,
                "alpha_blend": mat.alpha_blend,
            }
            for name, mat in MATERIALS.items()
            if name in set(material_assignments.values())
        },
        "assignments": material_assignments,
        "pending": {
            "texture_maps": "normal / AO / UV-mapped albedo maps pending the texture pipeline stage",
        },
    }

    return ReconstructionResult(
        scene_glb=glb,
        meters_per_px=scale,
        stats=stats,
        furniture=furniture_report,
        geometry_manifest=geometry_manifest,
        materials_manifest=materials_manifest,
    )
