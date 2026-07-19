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

    assert {"bed", "mattress", "wardrobe"} <= {p.name for p in placed}
    for p in placed:
        assert p.footprint.within(room), f"{p.name} pokes outside the room"

    # Collision rules apply to ground-level bases; stacked pieces (mattress,
    # blanket) share their base's footprint by design, elevated pieces
    # (mirror) clear the floor.
    stacked = {"mattress", "blanket"}
    bases = [p for p in placed if p.z0 == 0 and p.name not in stacked]
    for i, a in enumerate(bases):
        for b in bases[i + 1 :]:
            assert not a.footprint.intersects(b.footprint), f"{a.name} collides with {b.name}"

    # Large pieces keep full circulation clearance between each other.
    large = {p.name: p for p in bases if p.name in ("bed", "wardrobe", "study_table")}
    names = list(large)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert large[a].footprint.distance(large[b].footprint) >= 0.4


def test_furniture_skipped_when_room_too_small():
    from shapely.geometry import box

    from app.pipeline.furniture import place_furniture

    closet = box(0, 0, 0.4, 0.4)
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
    result = run_pipeline(plan_png, furniture_mode="detected")

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


def test_stage4_geometry_elements(plan_png):
    result = run_pipeline(plan_png)
    scene = trimesh.load(io.BytesIO(result.glb), file_type="glb")
    names = list(scene.geometry)

    # Door frames + lintels exist for each of the 2 detected openings.
    assert sum(1 for n in names if n.startswith("lintel_")) == 2
    assert sum(1 for n in names if n.startswith("door_frame_")) == 4
    # Skirting per room, parapet on the roof.
    assert sum(1 for n in names if n.startswith("skirting_")) >= 3
    assert any(n.startswith("roof_parapet_") for n in names)

    assert result.validation["no_floating_meshes"] is True
    assert result.validation["passed"] is True

    manifest = result.geometry
    assert manifest["stage"] == "geometry"
    assert manifest["node_counts"]["lintel"] == 2
    assert "no windows detected" in manifest["pending"]["window_frames_glass"]


def test_stage5_auto_mode_never_leaves_rooms_empty(plan_png):
    result = run_pipeline(plan_png)  # default: auto

    scene_json = result.furnishing
    assert scene_json["stage"] == "furnishing"
    assert scene_json["mode"] == "auto"

    by_room = {r["id"]: r["items"] for r in scene_json["rooms"]}
    assert all(len(items) > 0 for items in by_room.values()), "no room may be empty"

    # Rooms with drawn symbols keep exactly those (fidelity)...
    assert all(f["source"] == "detected" for f in by_room[0])  # living: drawn rect
    assert all(f["source"] == "detected" for f in by_room[2])  # bathroom: drawn circle
    # ...and the symbol-less bedroom is furnished from the catalog.
    bedroom_items = {f["item"] for f in by_room[1]}
    assert {"bed", "mattress", "wardrobe"} <= bedroom_items
    assert all(f["source"] == "generated" for f in by_room[1])

    # Door clearance: generated pieces stay clear of both door centers.
    mpp = result.stats["meters_per_px"]
    door_centers = [(o["center_px"][0] * mpp, -o["center_px"][1] * mpp) for o in result.openings]
    for f in by_room[1]:
        if f["item"] in ("mattress", "blanket"):
            continue
        px, py = f["position_m"]
        for ox, oy in door_centers:
            d = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
            assert d > 0.6, f"{f['item']} sits in a doorway (dist {d:.2f} m)"


def _glb_json(glb: bytes) -> dict:
    import json as _json
    import struct as _struct

    assert glb[:4] == b"glTF"
    jlen = _struct.unpack("<I", glb[12:16])[0]
    assert glb[16:20] == b"JSON"
    return _json.loads(glb[20 : 20 + jlen])


def test_stage6_pbr_materials_in_glb(plan_png):
    result = run_pipeline(plan_png)
    doc = _glb_json(result.glb)

    mats = {m["name"]: m for m in doc["materials"]}
    assert "wall_paint" in mats and "concrete" in mats
    for m in mats.values():
        pbr = m["pbrMetallicRoughness"]
        assert "baseColorFactor" in pbr
        assert 0.0 <= pbr["roughnessFactor"] <= 1.0
        assert 0.0 <= pbr["metallicFactor"] <= 1.0
    if "mirror" in mats:
        assert mats["mirror"]["pbrMetallicRoughness"]["metallicFactor"] == 1.0

    manifest = result.materials
    assert manifest["stage"] == "materials"
    assert manifest["workflow"] == "pbr_metallic_roughness"
    # Every scene node has an assignment; floors follow the room type.
    assert manifest["assignments"]["room_0_living_room"] == "marble_floor"
    assert manifest["assignments"]["room_1_master_bedroom"] == "wood_floor"
    assert manifest["assignments"]["room_2_bathroom"] == "ceramic_tile"
    assert "pending" in manifest and "texture_maps" in manifest["pending"]


def test_stage7_decor_present_but_bounded(plan_png):
    result = run_pipeline(plan_png)

    by_room: dict[int, list] = {}
    for f in result.furniture:
        by_room.setdefault(f["room_id"], []).append(f)

    # The symbol-less master bedroom gets furnished including bed dressing.
    bedroom_items = {f["item"] for f in by_room[1]}
    assert {"pillow_left", "pillow_right", "blanket"} <= bedroom_items
    # Never overdecorate: at most 5 decor pieces in any room.
    for room_id, items in by_room.items():
        decor_count = sum(1 for f in items if f.get("decor"))
        assert decor_count <= 5, f"room {room_id} overdecorated ({decor_count})"


def test_stage8_lights_embedded_in_glb(plan_png):
    result = run_pipeline(plan_png)
    doc = _glb_json(result.glb)

    assert "KHR_lights_punctual" in doc["extensionsUsed"]
    lights = doc["extensions"]["KHR_lights_punctual"]["lights"]
    # One sun + one ceiling light per room.
    assert len(lights) == 1 + len(result.rooms)
    assert lights[0]["type"] == "directional"
    assert all(l["type"] == "point" for l in lights[1:])

    light_nodes = [n for n in doc["nodes"] if "KHR_lights_punctual" in n.get("extensions", {})]
    assert len(light_nodes) == len(lights)
    # Ceiling lights sit just under the 3.0 m ceiling, inside the building.
    for n in light_nodes:
        if "translation" in n:
            x, y, z = n["translation"]
            assert 2.5 < y < 3.0
            assert 0 < x < 30 and 0 < z < 30

    manifest = result.lighting
    assert manifest["stage"] == "lighting"
    assert len(manifest["lights"]) == len(lights)
    assert "renderer_recommendations" in manifest

    # The lit GLB still round-trips through a loader.
    scene = trimesh.load(io.BytesIO(result.glb), file_type="glb")
    assert len(scene.geometry) > 0


def test_stage9_cameras_embedded_and_walkthrough(plan_png):
    result = run_pipeline(plan_png)
    doc = _glb_json(result.glb)

    cams = {c["name"]: c for c in doc["cameras"]}
    for expected in ("top_orthographic", "isometric_45", "bird_eye",
                     "front_elevation", "rear_elevation"):
        assert expected in cams, expected
    assert cams["top_orthographic"]["type"] == "orthographic"
    assert cams["isometric_45"]["type"] == "perspective"
    # One interior camera per room.
    assert sum(1 for n in cams if n.startswith("interior_")) == len(result.rooms)

    cam_nodes = [n for n in doc["nodes"] if "camera" in n]
    assert len(cam_nodes) == len(doc["cameras"])
    assert result.validation["cameras_embedded"] == len(doc["cameras"])

    walk = result.cameras["walkthrough"]
    depths = [w["door_depth"] for w in walk["waypoints"]]
    assert depths[0] == 0                      # starts at the entrance room
    assert depths == sorted(depths)            # ordered by door depth
    assert all(w["position"][1] == 1.55 for w in walk["waypoints"])


def test_stage11_validation_checks(plan_png):
    v = run_pipeline(plan_png).validation

    for check in ("no_mesh_overlaps", "no_duplicate_objects", "wall_height_ok",
                  "materials_assigned", "room_labels_ok", "no_floating_meshes",
                  "hierarchy_consistent", "accessibility_ok"):
        assert v[check] is True, check
    assert v["repairs"] == []
    assert v["lights_embedded"] == len(run_pipeline(plan_png).rooms) + 1
    assert "pending" in v["uv_maps"]
    assert v["passed"] is True


def test_stage11_auto_repair_removes_illegal_generated_piece(plan_png):
    """Fabricate a floating generated box in the GLB and verify repair removes
    it (and only it), while a detected object would be left alone."""
    import trimesh as tm

    from app.pipeline.types import ReconstructionResult
    from app.pipeline.validate import validate_and_repair
    from app.pipeline.graph import build_graph  # noqa: F401  (import sanity)

    result = run_pipeline(plan_png, furniture_mode="none")
    scene = tm.load(io.BytesIO(result.glb), file_type="glb")
    bad = tm.creation.box(extents=(0.5, 0.5, 0.5))
    bad.apply_translation((5.0, 25.0, 5.0))  # far above the roof: floating
    scene.add_geometry(bad, node_name="furniture_0_ghost_0", geom_name="furniture_0_ghost_0")
    glb = scene.export(file_type="glb")

    recon = ReconstructionResult(scene_glb=glb, meters_per_px=result.stats["meters_per_px"])
    # Rebuild a plan graph cheaply via the pipeline internals is overkill here;
    # reuse rooms from the result through a minimal PlanGraph.
    pg = _make_graph([(r["id"], 3.0, 3.0) for r in result.rooms],
                     [(a["a"], a["b"], a["opening"]) for a in result.adjacency])
    for room in pg.rooms:
        room.label = "bedroom"

    checks, repaired = validate_and_repair(
        recon, pg, furniture_sources={"generated": ["furniture_0_ghost_0"], "detected": []}
    )
    assert "furniture_0_ghost_0" in checks["repaired_nodes"]
    assert checks["no_floating_meshes"] is True
    assert repaired is not None
    rescene = tm.load(io.BytesIO(repaired), file_type="glb")
    assert "furniture_0_ghost_0" not in rescene.geometry


def test_stage12_export_formats(plan_png):
    import ifcopenshell
    import json as _json
    import tempfile
    from pathlib import Path

    from app.pipeline import exports

    glb = run_pipeline(plan_png).glb

    # Embedded glTF preserves lights and cameras from the GLB byte-for-byte.
    gltf = exports.glb_to_gltf_embedded(glb)
    doc = _json.loads(gltf)
    assert doc["buffers"][0]["uri"].startswith("data:application/octet-stream;base64,")
    assert "KHR_lights_punctual" in doc["extensionsUsed"]
    assert len(doc["cameras"]) >= 8

    # IFC round-trips through ifcopenshell with the right classes.
    ifc = exports.export_ifc(glb)
    assert ifc.startswith(b"ISO-10303-21")
    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
        tmp.write(ifc)
        name = tmp.name
    f = ifcopenshell.open(name)
    Path(name).unlink()
    assert len(f.by_type("IfcWall")) >= 2
    assert len(f.by_type("IfcSpace")) == 3
    assert len(f.by_type("IfcSlab")) >= 2
    assert len(f.by_type("IfcFurnishingElement")) > 0
    assert len(f.by_type("IfcBuildingStorey")) == 1

    # USDZ is a valid zip package.
    usdz = exports.export_usdz(glb)
    assert usdz[:2] == b"PK"
    assert len(usdz) > 5000
