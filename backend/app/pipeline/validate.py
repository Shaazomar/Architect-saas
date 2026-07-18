"""Stage 7 — sanity checks on the reconstructed model.

Each check is reported individually so the client can show precise warnings;
``passed`` is the conjunction of the hard checks only.
"""
from __future__ import annotations

import io
from typing import Any

import networkx as nx
import numpy as np
import trimesh

from .types import PlanGraph, ReconstructionResult


def validate(recon: ReconstructionResult, plan_graph: PlanGraph) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    scene = trimesh.load(io.BytesIO(recon.scene_glb), file_type="glb")
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]

    checks["geometry_present"] = len(meshes) > 0
    checks["no_degenerate_meshes"] = all(m.faces.shape[0] > 0 and m.area > 0 for m in meshes)
    checks["walls_watertight"] = all(
        m.is_watertight for name, m in scene.geometry.items() if name.startswith("wall_")
    )

    # No floating meshes: every mesh must rest on/below floor level or have a
    # bounding box touching another grounded mesh (lintels touch walls, upper
    # cabinets sit within the wall envelope, stacked pieces touch their base).
    named = [(n, m) for n, m in scene.geometry.items() if isinstance(m, trimesh.Trimesh)]
    pad = 0.05
    grounded = {n for n, m in named if m.bounds[0][1] <= pad}
    changed = True
    while changed:
        changed = False
        for n, m in named:
            if n in grounded:
                continue
            lo, hi = m.bounds[0] - pad, m.bounds[1] + pad
            for gn, gm in named:
                if gn not in grounded:
                    continue
                glo, ghi = gm.bounds
                if np.all(lo <= ghi) and np.all(glo <= hi):
                    grounded.add(n)
                    changed = True
                    break
    floating = sorted(n for n, _ in named if n not in grounded)
    checks["no_floating_meshes"] = not floating
    checks["floating_meshes"] = floating

    if scene.bounds is not None:
        extent = (scene.bounds[1] - scene.bounds[0]).max()
        checks["plausible_scale"] = bool(1.0 <= extent <= 500.0)
        checks["extent_m"] = float(extent)
    else:
        checks["plausible_scale"] = False

    g = plan_graph.graph
    checks["room_count"] = g.number_of_nodes()
    checks["rooms_all_reachable"] = (
        g.number_of_nodes() > 0
        and nx.is_connected(g)
        if g.number_of_nodes() > 0
        else False
    )
    checks["isolated_rooms"] = [n for n in g.nodes if g.degree(n) == 0]

    hard = ("geometry_present", "no_degenerate_meshes", "plausible_scale", "no_floating_meshes")
    checks["passed"] = all(bool(checks[k]) for k in hard)
    return checks
