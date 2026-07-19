"""Stage 11 — Validation AI: check everything, repair what is safely fixable.

Checks: geometry presence/degeneracy, watertight walls, plausible scale and
wall height, no floating meshes, no overlaps or duplicates among *generated*
furniture, full material coverage, room labels, reachability.

Auto-repair policy: only generated furniture may be removed (an overlapping
or floating catalog piece is a placement bug). Detected objects, walls and
structure are never deleted — they represent the drawing, and silently
"repairing" them would falsify the reconstruction; they surface as failed
checks instead. Every repair is recorded in the report.
"""
from __future__ import annotations

import io
from typing import Any

import networkx as nx
import numpy as np
import trimesh

from ..config import settings
from .types import PlanGraph, ReconstructionResult

# Pieces allowed to overlap others by design: overlays, stacked pieces,
# elevated pieces, and glass panels that sit against trays.
_OVERLAP_EXEMPT = (
    "rug", "prayer_mat", "mattress", "blanket", "pillow", "books", "microwave",
    "fruit_bowl", "coffee_machine", "knife_stand", "centerpiece", "mirror",
    "upper_cabinet", "wall_art", "led_strip", "towel", "glass_partition",
)


def _xz_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    (ax0, _, az0), (ax1, _, az1) = a.bounds
    (bx0, _, bz0), (bx1, _, bz1) = b.bounds
    dx = min(ax1, bx1) - max(ax0, bx0)
    dz = min(az1, bz1) - max(az0, bz0)
    return dx * dz if dx > 0 and dz > 0 else 0.0


def _floating(named: list[tuple[str, trimesh.Trimesh]], pad: float = 0.05) -> list[str]:
    grounded = {n for n, m in named if m.bounds[0][1] <= pad}
    changed = True
    while changed:
        changed = False
        for n, m in named:
            if n in grounded:
                continue
            lo, hi = m.bounds[0] - pad, m.bounds[1] + pad
            for gn, gm in named:
                if gn in grounded:
                    glo, ghi = gm.bounds
                    if np.all(lo <= ghi) and np.all(glo <= hi):
                        grounded.add(n)
                        changed = True
                        break
    return sorted(n for n, _ in named if n not in grounded)


def validate_and_repair(
    recon: ReconstructionResult,
    plan_graph: PlanGraph,
    assignments: dict[str, str] | None = None,
    furniture_sources: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    checks: dict[str, Any] = {}
    repairs: list[str] = []

    scene = trimesh.load(io.BytesIO(recon.scene_glb), file_type="glb")
    generated = set((furniture_sources or {}).get("generated", []))
    removed_all: set[str] = set()

    def named() -> list[tuple[str, trimesh.Trimesh]]:
        return [(n, m) for n, m in scene.geometry.items() if isinstance(m, trimesh.Trimesh)]

    def removable(name: str) -> bool:
        return name in generated

    # --- duplicates (identical centroid + volume) among furniture ---
    furn = [(n, m) for n, m in named() if n.startswith("furniture_")]
    dupes: list[str] = []
    for i, (na, ma) in enumerate(furn):
        for nb, mb in furn[i + 1 :]:
            if nb in dupes:
                continue
            if (np.linalg.norm(ma.centroid - mb.centroid) < 1e-3
                    and abs(ma.volume - mb.volume) < 1e-6):
                dupes.append(nb)
    for name in dupes:
        if removable(name):
            scene.delete_geometry(name)
            removed_all.add(name)
            repairs.append(f"removed duplicate generated piece {name}")
    checks["no_duplicate_objects"] = not [d for d in dupes if not removable(d)]

    # --- overlaps among generated ground-level furniture ---
    overlaps: list[tuple[str, str, float]] = []
    ground = [
        (n, m) for n, m in named()
        if n.startswith("furniture_") and m.bounds[0][1] < 1.2
        and not any(k in n for k in _OVERLAP_EXEMPT)
    ]
    removed: set[str] = set()
    for i, (na, ma) in enumerate(ground):
        for nb, mb in ground[i + 1 :]:
            if nb in removed or na in removed:
                continue
            area = _xz_overlap(ma, mb)
            if area > 0.02:
                overlaps.append((na, nb, round(area, 3)))
                if removable(nb):
                    scene.delete_geometry(nb)
                    removed.add(nb)
                    removed_all.add(nb)
                    repairs.append(f"removed overlapping generated piece {nb} (overlap {area:.2f} m² with {na})")
    checks["no_mesh_overlaps"] = not [o for o in overlaps if o[1] not in removed]
    checks["overlaps_found"] = [list(o) for o in overlaps]

    # --- floating meshes; floating *generated* furniture is removed ---
    floating = _floating(named())
    for name in list(floating):
        if removable(name):
            scene.delete_geometry(name)
            floating.remove(name)
            removed_all.add(name)
            repairs.append(f"removed floating generated piece {name}")
    checks["no_floating_meshes"] = not floating
    checks["floating_meshes"] = floating

    meshes = [m for _, m in named()]
    checks["geometry_present"] = len(meshes) > 0
    checks["no_degenerate_meshes"] = all(m.faces.shape[0] > 0 and m.area > 0 for m in meshes)
    checks["walls_watertight"] = all(
        m.is_watertight and m.volume > 0
        for n, m in named() if n.startswith("wall_")
    )

    wall_tops = [m.bounds[1][1] for n, m in named() if n.startswith("wall_")]
    checks["wall_height_ok"] = bool(
        wall_tops and abs(max(wall_tops) - settings.wall_height_m) < 0.05
    )

    if scene.bounds is not None:
        extent = (scene.bounds[1] - scene.bounds[0]).max()
        checks["plausible_scale"] = bool(1.0 <= extent <= 500.0)
        checks["extent_m"] = float(extent)
    else:
        checks["plausible_scale"] = False

    if assignments is not None:
        unassigned = [n for n, _ in named() if n not in assignments]
        checks["materials_assigned"] = not unassigned
        checks["unassigned_nodes"] = unassigned
    checks["uv_maps"] = "pending texture pipeline (factor-based PBR in use)"

    checks["room_labels_ok"] = all(r.label and r.label != "room" for r in plan_graph.rooms)

    g = plan_graph.graph
    checks["room_count"] = g.number_of_nodes()
    checks["rooms_all_reachable"] = (
        nx.is_connected(g) if g.number_of_nodes() > 0 else False
    )
    checks["isolated_rooms"] = [n for n in g.nodes if g.degree(n) == 0]

    checks["repairs"] = repairs
    checks["repaired_nodes"] = sorted(removed_all)

    hard = (
        "geometry_present", "no_degenerate_meshes", "plausible_scale",
        "no_floating_meshes", "no_mesh_overlaps", "no_duplicate_objects",
        "wall_height_ok", "room_labels_ok",
    )
    checks["passed"] = all(bool(checks[k]) for k in hard)

    repaired_glb: bytes | None = None
    if repairs:
        exported = scene.export(file_type="glb")
        repaired_glb = exported.encode() if isinstance(exported, str) else exported
    return checks, repaired_glb


def validate(recon: ReconstructionResult, plan_graph: PlanGraph) -> dict[str, Any]:
    """Back-compat wrapper: checks only, no repair."""
    checks, _ = validate_and_repair(recon, plan_graph)
    return checks
