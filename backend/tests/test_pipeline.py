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


def test_reports_and_furniture_in_pipeline_result(plan_png):
    result = run_pipeline(plan_png)

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
