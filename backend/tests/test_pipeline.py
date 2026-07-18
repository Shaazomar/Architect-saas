from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
import trimesh

from app.pipeline.preprocess import PlanImageError, preprocess
from app.pipeline.runner import run_pipeline


def test_full_pipeline_reconstructs_three_rooms(plan_png):
    result = run_pipeline(plan_png)

    assert len(result.rooms) == 3
    assert result.validation["passed"] is True
    assert result.validation["rooms_all_reachable"] is True

    # Every partition in the sample has a door gap, so all adjacencies are openings.
    assert len(result.adjacency) >= 2
    assert all(edge["opening"] for edge in result.adjacency)

    # The GLB must load back as real geometry at a plausible building scale.
    scene = trimesh.load(io.BytesIO(result.glb), file_type="glb")
    assert len(scene.geometry) > 0
    extent = (scene.bounds[1] - scene.bounds[0]).max()
    assert 5.0 < extent < 100.0

    labels = {room["label"] for room in result.rooms}
    assert "living_room" in labels
    assert all(room["area_m2"] > 1.0 for room in result.rooms)


def test_rejects_undecodable_bytes():
    with pytest.raises(PlanImageError):
        preprocess(b"not an image at all" * 100)


def test_rejects_blank_page():
    blank = np.full((800, 800), 255, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    with pytest.raises(PlanImageError):
        run_pipeline(buf.tobytes())


def test_rejects_tiny_image():
    dot = np.zeros((10, 10), np.uint8)
    ok, buf = cv2.imencode(".png", dot)
    assert ok
    with pytest.raises(PlanImageError):
        preprocess(buf.tobytes())


def test_explicit_scale_overrides_estimate(plan_png):
    result = run_pipeline(plan_png, meters_per_px=0.01)
    assert result.stats["meters_per_px"] == 0.01


def test_furniture_is_placed_legally():
    from shapely.geometry import box

    from app.pipeline.furniture import place_furniture

    room = box(0, 0, 5.0, 4.0)
    placed = place_furniture(0, "bedroom", room)

    assert {p.name for p in placed} == {"bed", "wardrobe", "desk"}
    for p in placed:
        assert p.footprint.within(room), f"{p.name} pokes outside the room"
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            assert not a.footprint.intersects(b.footprint), f"{a.name} collides with {b.name}"
            assert a.footprint.distance(b.footprint) >= 0.4  # circulation preserved


def test_furniture_skipped_when_room_too_small():
    from shapely.geometry import box

    from app.pipeline.furniture import place_furniture

    closet = box(0, 0, 0.6, 0.6)
    assert place_furniture(0, "bedroom", closet) == []


def test_openings_detected_from_wall_gaps(plan_png):
    result = run_pipeline(plan_png)

    # The sample plan has exactly two door gaps (one per partition).
    assert len(result.openings) == 2
    for op in result.openings:
        assert 1.0 <= op["width_m"] <= 2.2
        assert len(op["rooms"]) == 2
        assert op["confidence"] > 0
    assert result.validation["opening_count"] == 2
    assert result.validation["window_count"] == 0
    assert any("wall thickness" in w for w in result.validation["warnings"])


def test_detected_furniture_matches_drawn_symbols(plan_png):
    """Fidelity mandate: only the two symbols drawn in the sample (a rectangle
    in the living room, a circle in the bathroom) become objects — the text
    labels and dimension lines must not."""
    result = run_pipeline(plan_png)  # default mode is "detected"

    assert len(result.furniture) == 2
    assert all(f["source"] == "detected" for f in result.furniture)
    by_room = {f["room_label"]: f for f in result.furniture}
    assert set(by_room) == {"living_room", "bathroom"}
    assert by_room["bathroom"]["item"] == "round_table"  # the drawn circle
    assert all(0 < f["confidence"] <= 1 for f in result.furniture)

    rooms_in_graph = result.scene_graph["building"]["rooms"]
    assert sum(len(r["objects"]) for r in rooms_in_graph) == 2


def test_generated_furniture_is_explicit_opt_in(plan_png):
    result = run_pipeline(plan_png, furniture_mode="generated")
    assert len(result.furniture) >= 6
    assert all(f["source"] == "generated" for f in result.furniture)

    none = run_pipeline(plan_png, furniture_mode="none")
    assert none.furniture == []

    with pytest.raises(ValueError):
        run_pipeline(plan_png, furniture_mode="invent")


def test_reports_and_furniture_in_pipeline_result(plan_png):
    result = run_pipeline(plan_png, furniture_mode="generated")

    schedule = result.reports["room_schedule"]
    assert len(schedule) == 3
    assert all(r["perimeter_m"] > 0 for r in schedule)

    materials = result.reports["materials"]
    assert materials["concrete_total_m3"] > 0
    assert materials["paint_area_m2"] > 0

    cost = result.reports["cost_estimate"]
    assert cost["total"] > 0 and cost["currency"] == "INR"
    assert "disclaimer" in cost

    assert len(result.furniture) > 0
    # Furniture nodes really exist in the exported scene.
    scene = trimesh.load(io.BytesIO(result.glb), file_type="glb")
    furn_nodes = [n for n in scene.geometry if n.startswith("furniture_")]
    assert len(furn_nodes) == len(result.furniture)
    assert any(n.startswith("roof_") for n in scene.geometry)


def test_stage1_analysis_is_detection_only(plan_png):
    analysis = run_pipeline(plan_png).analysis

    assert analysis["stage"] == "floor_plan_analysis"
    assert analysis["scale"]["source"] == "wall_thickness_assumption"
    assert len(analysis["walls"]) >= 1
    assert all(len(w["polygon_px"]) >= 4 for w in analysis["walls"])
    assert len(analysis["rooms"]) == 3
    assert len(analysis["doors"]) == 2
    assert len(analysis["symbols"]) == 2

    # Never guess: undetectable classes are explicitly empty + pending.
    for key in ("windows", "stairs", "columns", "dimensions", "text"):
        assert analysis[key]["items"] == []
        assert "pending" in analysis[key]["status"]
    assert analysis["north_arrow"]["detected"] is False


def _make_graph(specs, edges):
    """specs: (id, w_m, h_m); mpp=1 so px == m."""
    import networkx as nx
    from shapely.geometry import box

    from app.pipeline.types import PlanGraph, Room

    rooms = [
        Room(id=i, polygon=box(0, j * 100, w, j * 100 + h), area_px=w * h)
        for j, (i, w, h) in enumerate(specs)
    ]
    g = nx.Graph()
    g.add_nodes_from(r.id for r in rooms)
    for a, b, opening in edges:
        g.add_edge(a, b, opening=opening)
    return PlanGraph(graph=g, rooms=rooms)


def test_stage2_classifier_uses_size_and_door_evidence():
    from app.pipeline.classify import classify_rooms

    pg = _make_graph(
        [(0, 5.0, 4.0), (1, 3.2, 3.0), (2, 2.2, 2.0), (3, 4.0, 3.5)],
        [(0, 1, True), (0, 3, True), (3, 2, True)],
    )
    classify_rooms(pg, meters_per_px=1.0)
    labels = {r.id: r.label for r in pg.rooms}

    assert labels[0] == "living_room"          # largest (20 m2)
    assert labels[1] == "kitchen"              # 9.6 m2, door to living
    assert labels[2] == "bathroom"             # 4.4 m2, no door to living
    assert labels[3] == "master_bedroom"       # 14 m2 private room
    for r in pg.rooms:
        assert 0 < r.label_confidence <= 1
        assert r.evidence, f"room {r.id} has no evidence trail"


def test_stage2_toilet_off_living_and_store():
    from app.pipeline.classify import classify_rooms

    pg = _make_graph(
        [(0, 6.0, 4.0), (1, 2.0, 2.5), (2, 1.4, 1.5)],
        [(0, 1, True), (0, 2, False)],
    )
    classify_rooms(pg, meters_per_px=1.0)
    labels = {r.id: r.label for r in pg.rooms}
    assert labels[1] == "common_toilet"        # wet-size, door to living
    assert labels[2] == "store"                # under 3 m2


def test_stage2_rooms_artifact(plan_png):
    detail = run_pipeline(plan_png).rooms_detail

    assert detail["stage"] == "room_detection"
    rooms = detail["rooms"]
    assert len(rooms) == 3
    names = {r["name"] for r in rooms}
    assert "living_room" in names
    for r in rooms:
        assert len(r["polygon_px"]) >= 4
        assert len(r["center_px"]) == 2 and len(r["center_m"]) == 2
        assert r["area_m2"] > 0 and r["perimeter_m"] > 0
        assert r["windows"]["status"] == "pending_ml_detector"
        assert isinstance(r["doors"], list) and isinstance(r["adjacent_rooms"], list)
        assert 0 < r["confidence"] <= 1 and r["evidence"]
    # Every detected door belongs to exactly two rooms' door lists.
    door_refs = [d for r in rooms for d in r["doors"]]
    assert sorted(door_refs) == [0, 0, 1, 1]


def test_stage3_building_graph(plan_png):
    bg = run_pipeline(plan_png).building_graph

    assert bg["stage"] == "building_graph"
    assert len(bg["nodes"]) == 3

    zones_by_room = {n["id"]: n["zone"] for n in bg["nodes"]}
    assert zones_by_room[0] == "public"        # living_room
    assert zones_by_room[1] == "private"       # master_bedroom
    assert zones_by_room[2] == "service"       # bathroom

    # Door edges must reference detected openings; the third adjacency
    # (living-bathroom share a wall but no detected opening) stays adjacency.
    door_edges = [e for e in bg["edges"] if e["type"] == "door"]
    adj_edges = [e for e in bg["edges"] if e["type"] == "adjacency"]
    assert len(door_edges) == 2
    assert all(e["opening_id"] is not None and e["width_m"] > 0 for e in door_edges)
    assert all(e["opening_id"] is None for e in adj_edges)

    acc = bg["accessibility"]
    assert acc["entrance_room"] == 0           # assumed public room (living)
    assert acc["entrance_source"] == "assumed_public_room"
    assert acc["all_rooms_reachable"] is True
    assert acc["door_depth_from_entrance"]["0"] == 0
    assert bg["windows"]["status"] == "pending_ml_detector"
    assert set(bg["zones"]) == {"public", "private", "service"}
    assert bg["hierarchy"]["building"]["floors"][0]["zones"]["public"] == [0]
