"""Stage 3 — Semantic Building Graph.

Rooms become nodes, doors become edges, walls become boundaries; windows
join as openings when the ML detector lands. On top of the raw topology the
graph stores what an architect reads out of it: zoning (public / private /
service / circulation), the room hierarchy, and accessibility — how every
room is reached from the entrance.

An edge is typed "door" only when Stage 1 detected a matching opening; a
wall shared without a detected opening stays an "adjacency" edge (with the
raw-gap hint preserved), never promoted to a door.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from .types import PlanGraph

_ZONES = {
    "living_room": "public",
    "dining": "public",
    "balcony": "public",
    "terrace": "public",
    "hallway": "circulation",
    "bedroom": "private",
    "master_bedroom": "private",
    "study": "private",
    "pooja_room": "private",
    "kitchen": "service",
    "bathroom": "service",
    "common_toilet": "service",
    "utility": "service",
    "laundry": "service",
    "store": "service",
    "garage": "service",
}


def build_semantic_graph(
    plan_graph: PlanGraph,
    openings_out: list[dict[str, Any]],
    meters_per_px: float,
) -> dict[str, Any]:
    g = plan_graph.graph
    rooms = plan_graph.rooms

    opening_by_pair = {
        frozenset(o["rooms"]): o for o in openings_out if "exterior" not in o["rooms"]
    }
    exterior_openings = [o for o in openings_out if "exterior" in o["rooms"]]

    nodes = [
        {
            "id": r.id,
            "name": r.label,
            "zone": _ZONES.get(r.label, "unzoned"),
            "area_m2": round(r.area_px * meters_per_px**2, 2),
            "degree": g.degree(r.id) if r.id in g else 0,
        }
        for r in rooms
    ]

    edges = []
    door_graph = nx.Graph()
    door_graph.add_nodes_from(r.id for r in rooms)
    for a, b, data in g.edges(data=True):
        opening = opening_by_pair.get(frozenset((a, b)))
        if opening is not None:
            edges.append(
                {"a": a, "b": b, "type": "door", "opening_id": opening["id"],
                 "width_m": opening["width_m"]}
            )
            door_graph.add_edge(a, b)
        else:
            edges.append(
                {"a": a, "b": b, "type": "adjacency", "opening_id": None,
                 "raw_gap_evidence": bool(data.get("opening", False))}
            )

    # Entrance: a detected exterior door wins; otherwise assume the living
    # area (stated as an assumption, per the no-guessing rule).
    if exterior_openings:
        entrance_room = next(r for r in exterior_openings[0]["rooms"] if r != "exterior")
        entrance_source = "exterior_opening"
    else:
        public = [n for n in nodes if n["zone"] == "public"]
        entrance_room = (public[0]["id"] if public else nodes[0]["id"]) if nodes else None
        entrance_source = "assumed_public_room"

    depth = (
        nx.single_source_shortest_path_length(door_graph, entrance_room)
        if entrance_room is not None and entrance_room in door_graph
        else {}
    )
    unreachable = [r.id for r in rooms if r.id not in depth]

    zones: dict[str, list[int]] = {"public": [], "private": [], "service": [], "circulation": [], "unzoned": []}
    for n in nodes:
        zones[n["zone"]].append(n["id"])

    return {
        "stage": "building_graph",
        "nodes": nodes,
        "edges": edges,
        "boundaries": {
            "walls_ref": "analysis.json#/walls",
            "room_perimeters_m": {
                str(r.id): round(r.polygon.exterior.length * meters_per_px, 2) for r in rooms
            },
        },
        "windows": {"items": [], "status": "pending_ml_detector"},
        "zones": {k: v for k, v in zones.items() if v},
        "hierarchy": {
            "building": {
                "floors": [
                    {"level": 0, "zones": {k: v for k, v in zones.items() if v}}
                ]
            }
        },
        "accessibility": {
            "entrance_room": entrance_room,
            "entrance_source": entrance_source,
            "all_rooms_reachable": not unreachable,
            "unreachable_rooms": unreachable,
            "door_depth_from_entrance": {str(k): v for k, v in depth.items()},
        },
        "circulation": {
            "circulation_rooms": zones["circulation"],
            "max_door_depth": max(depth.values()) if depth else None,
        },
    }
