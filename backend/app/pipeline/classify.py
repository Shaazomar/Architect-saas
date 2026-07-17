"""Stage 5 — room classification.

Heuristic MVP: rank rooms by area and degree in the connectivity graph.
The contract (PlanGraph in, labelled rooms out) is exactly what the future
GraphSAGE/GCN classifier will implement; OCR room-name extraction slots in
here as well and overrides any inferred label.
"""
from __future__ import annotations

from .types import PlanGraph


def classify_rooms(plan_graph: PlanGraph) -> PlanGraph:
    if not plan_graph.rooms:
        return plan_graph

    by_area = sorted(plan_graph.rooms, key=lambda r: r.area_px, reverse=True)
    g = plan_graph.graph

    for rank, room in enumerate(by_area):
        degree = g.degree(room.id) if room.id in g else 0
        if rank == 0:
            room.label = "living_room"
        elif degree >= 3:
            room.label = "hallway"
        elif rank == len(by_area) - 1 and len(by_area) > 2:
            room.label = "bathroom"
        else:
            room.label = "bedroom"
        g.nodes[room.id]["label"] = room.label

    return plan_graph
