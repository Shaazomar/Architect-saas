"""Stage 2 — Room Detector (licensed-architect heuristics).

Evidence-based classification over the room graph. Each rule cites its
evidence and yields a confidence; a room that matches nothing keeps a weak
generic label rather than a confident wrong one. The contract (PlanGraph in,
labelled rooms out) is the same one the future GraphSAGE/GCN classifier and
the OCR label reader will implement — OCR text, when present, overrides any
inference here.

Vocabulary (subset fires only when the geometry supports it): living_room,
kitchen, master_bedroom, bedroom, study, bathroom, common_toilet, store,
hallway. Balcony/terrace/garden need exterior-context detection (ML stage).
"""
from __future__ import annotations

from .types import PlanGraph, Room

# Indian-residential size priors, m²
_STORE_MAX = 3.0
_BATH_MAX = 6.5
_KITCHEN_MIN, _KITCHEN_MAX = 6.5, 14.0
_MASTER_MIN = 12.0
_STUDY_MAX = 9.0
_HALLWAY_ASPECT = 2.6


def _aspect(room: Room) -> float:
    minx, miny, maxx, maxy = room.polygon.bounds
    w, h = maxx - minx, maxy - miny
    return max(w, h) / max(1e-6, min(w, h))


def _assign(room: Room, graph, label: str, confidence: float, evidence: list[str]) -> None:
    room.label = label
    room.label_confidence = round(confidence, 2)
    room.evidence = evidence
    if room.id in graph:
        graph.nodes[room.id]["label"] = label


def classify_rooms(plan_graph: PlanGraph, meters_per_px: float) -> PlanGraph:
    rooms, g = plan_graph.rooms, plan_graph.graph
    if not rooms:
        return plan_graph

    area = {r.id: r.area_px * meters_per_px**2 for r in rooms}
    by_area = sorted(rooms, key=lambda r: area[r.id], reverse=True)

    def door_neighbors(rid: int) -> set[int]:
        if rid not in g:
            return set()
        return {n for n in g.neighbors(rid) if g.edges[rid, n].get("opening")}

    living = by_area[0]
    living_conf = 0.6
    living_ev = [f"largest room ({area[living.id]:.0f} m²)"]
    if g.number_of_nodes() > 1 and g.degree(living.id) == max(dict(g.degree()).values()):
        living_conf += 0.1
        living_ev.append("most connected room")
    _assign(living, g, "living_room", living_conf, living_ev)

    unassigned: list[Room] = []
    for room in by_area[1:]:
        a = area[room.id]
        if a < _STORE_MAX:
            _assign(room, g, "store", 0.5, [f"very small ({a:.1f} m²)"])
        elif a < _BATH_MAX:
            if living.id in door_neighbors(room.id):
                _assign(room, g, "common_toilet", 0.55,
                        [f"wet-area size ({a:.1f} m²)", "door opens to living area"])
            else:
                _assign(room, g, "bathroom", 0.55, [f"wet-area size ({a:.1f} m²)"])
        elif _aspect(room) > _HALLWAY_ASPECT and g.degree(room.id) >= 2:
            _assign(room, g, "hallway", 0.5,
                    [f"elongated (aspect {_aspect(room):.1f})", f"connects {g.degree(room.id)} rooms"])
        else:
            unassigned.append(room)

    # One kitchen: the smallest kitchen-sized room with a door to the living area.
    kitchen_candidates = [
        r for r in unassigned
        if _KITCHEN_MIN <= area[r.id] <= _KITCHEN_MAX and living.id in door_neighbors(r.id)
    ]
    if kitchen_candidates:
        kitchen = min(kitchen_candidates, key=lambda r: area[r.id])
        _assign(kitchen, g, "kitchen", 0.5,
                [f"kitchen-sized ({area[kitchen.id]:.1f} m²)", "door opens to living area"])
        unassigned.remove(kitchen)

    # Remaining rooms are the bedroom pool, largest first.
    for rank, room in enumerate(sorted(unassigned, key=lambda r: area[r.id], reverse=True)):
        a = area[room.id]
        if rank == 0 and a >= _MASTER_MIN:
            _assign(room, g, "master_bedroom", 0.55, [f"largest bedroom ({a:.0f} m²)"])
        elif a < _STUDY_MAX and len(unassigned) >= 3:
            _assign(room, g, "study", 0.45, [f"small private room ({a:.1f} m²)"])
        else:
            _assign(room, g, "bedroom", 0.5, [f"private room ({a:.0f} m²)"])

    # A dwelling with 3+ rooms almost always has a wet area; if size rules
    # found none, mark the smallest non-living room — weakly.
    labels = {r.label for r in rooms}
    if len(rooms) > 2 and not labels & {"bathroom", "common_toilet"}:
        smallest = by_area[-1]
        if smallest.id != living.id:
            _assign(smallest, g, "bathroom", 0.4,
                    [f"smallest room ({area[smallest.id]:.0f} m²)", "size atypical — weak inference"])

    return plan_graph
