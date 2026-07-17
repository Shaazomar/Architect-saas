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
