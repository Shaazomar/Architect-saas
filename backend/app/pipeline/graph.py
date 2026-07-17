"""Stage 4 — room-connectivity graph.

Two rooms are adjacent when their polygons come within ~1.5 wall thicknesses
of each other. An adjacency counts as an *opening* (door/passage) when the
un-sealed free space of the drawing connects the two rooms — i.e. there is a
gap in the wall between them.

This graph is the input contract for the future GNN room classifier.
"""
from __future__ import annotations

import cv2
import networkx as nx
import numpy as np

from .types import DetectedStructure, PlanGraph, PreprocessedPlan, VectorPlan


def _openings_components(plan: PreprocessedPlan, structure: DetectedStructure) -> np.ndarray:
    """Connected components of raw (un-sealed) free space."""
    free = cv2.bitwise_not(structure.wall_mask)
    _, labels = cv2.connectedComponents(free)
    return labels


def build_graph(plan: PreprocessedPlan, structure: DetectedStructure, vector: VectorPlan) -> PlanGraph:
    g = nx.Graph()
    open_labels = _openings_components(plan, structure)

    room_open_ids: list[set[int]] = []
    for room in vector.rooms:
        g.add_node(room.id, area_px=room.area_px)
        cx, cy = room.polygon.representative_point().coords[0]
        x = int(np.clip(round(cx), 0, plan.width - 1))
        y = int(np.clip(round(cy), 0, plan.height - 1))
        room_open_ids.append({int(open_labels[y, x])})

    reach = vector.wall_thickness_px * 1.5
    for a in vector.rooms:
        for b in vector.rooms:
            if b.id <= a.id:
                continue
            if a.polygon.distance(b.polygon) > reach:
                continue
            # Same raw free-space component => a hole in the shared wall.
            opening = bool(room_open_ids[a.id] & room_open_ids[b.id])
            g.add_edge(a.id, b.id, opening=opening)

    return PlanGraph(graph=g, rooms=vector.rooms)
