from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture()
def client(isolated_data_dir):
    with TestClient(app) as c:
        yield c


def test_upload_and_full_job_lifecycle(client, plan_png):
    resp = client.post("/api/v1/plans", files={"file": ("plan.png", plan_png, "image/png")})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # TestClient runs background tasks before returning, so the job is finished.
    status = client.get(f"/api/v1/jobs/{job_id}").json()
    assert status["status"] == "done"
    assert len(status["result"]["rooms"]) == 3
    assert status["result"]["validation"]["passed"] is True

    model = client.get(f"/api/v1/jobs/{job_id}/model.glb")
    assert model.status_code == 200
    assert model.headers["content-type"] == "model/gltf-binary"
    assert model.content[:4] == b"glTF"


def test_rejects_non_image_upload(client):
    resp = client.post("/api/v1/plans", files={"file": ("evil.png", b"<script>alert(1)</script>", "image/png")})
    assert resp.status_code == 415


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    big = b"\x89PNG\r\n\x1a\n" + b"\x00" * 2048
    resp = client.post("/api/v1/plans", files={"file": ("big.png", big, "image/png")})
    assert resp.status_code == 413


def test_unknown_and_traversal_job_ids_are_404(client):
    assert client.get("/api/v1/jobs/no-such-job").status_code == 404
    assert client.get("/api/v1/jobs/..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/api/v1/jobs/no-such-job/model.glb").status_code == 404


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-key")
    assert client.get("/api/v1/jobs/x").status_code == 401
    assert client.get("/api/v1/jobs/x", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/jobs/x", headers={"X-API-Key": "secret-key"}).status_code == 404
    assert client.get("/health").status_code == 200  # health stays public


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"


def test_multi_format_export(client, plan_png):
    job_id = client.post(
        "/api/v1/plans", files={"file": ("plan.png", plan_png, "image/png")}
    ).json()["job_id"]

    status = client.get(f"/api/v1/jobs/{job_id}").json()
    assert status["status"] == "done"
    assert len(status["result"]["furniture"]) > 0
    assert status["result"]["reports"]["cost_estimate"]["total"] > 0

    for fmt in ("glb", "obj", "stl", "ply"):
        resp = client.get(f"/api/v1/jobs/{job_id}/model.{fmt}")
        assert resp.status_code == 200, fmt
        assert len(resp.content) > 500, fmt
    assert client.get(f"/api/v1/jobs/{job_id}/model.exe").status_code == 404
    assert client.get(f"/api/v1/jobs/{job_id}/model.%2e%2e").status_code == 404


def test_analysis_artifact_endpoint(client, plan_png):
    job_id = client.post(
        "/api/v1/plans", files={"file": ("plan.png", plan_png, "image/png")}
    ).json()["job_id"]

    resp = client.get(f"/api/v1/jobs/{job_id}/analysis.json")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "floor_plan_analysis"
    assert len(body["doors"]) == 2
    assert client.get("/api/v1/jobs/nope/analysis.json").status_code == 404

    rooms = client.get(f"/api/v1/jobs/{job_id}/rooms.json")
    assert rooms.status_code == 200
    assert rooms.json()["stage"] == "room_detection"

    graph = client.get(f"/api/v1/jobs/{job_id}/graph.json")
    assert graph.status_code == 200
    assert graph.json()["stage"] == "building_graph"

    assert client.get(f"/api/v1/jobs/{job_id}/secrets.json").status_code == 404


def test_stage10_renders_through_api(client, plan_png, monkeypatch):
    monkeypatch.setattr(settings, "renders_enabled", True)
    monkeypatch.setattr(settings, "render_width", 320)
    monkeypatch.setattr(settings, "render_height", 180)

    job_id = client.post(
        "/api/v1/plans", files={"file": ("plan.png", plan_png, "image/png")}
    ).json()["job_id"]

    status = client.get(f"/api/v1/jobs/{job_id}").json()
    renders = status["result"]["renders"]
    assert renders["status"] == "ok", renders
    names = {v["name"] for v in renders["views"]}
    assert {"top_orthographic", "isometric_45", "front_elevation"} <= names

    png = client.get(f"/api/v1/jobs/{job_id}/renders/top_orthographic.png")
    assert png.status_code == 200
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png.content) > 1000

    assert client.get(f"/api/v1/jobs/{job_id}/renders/../evil.png").status_code == 404
    assert client.get(f"/api/v1/jobs/{job_id}/renders/nope.png").status_code == 404
    manifest = client.get(f"/api/v1/jobs/{job_id}/renders.json")
    assert manifest.status_code == 200 and manifest.json()["status"] == "ok"

    # Stage 12: the walkthrough MP4 was rendered and is served.
    assert manifest.json()["walkthrough"]["file"] == "walkthrough.mp4"
    mp4 = client.get(f"/api/v1/jobs/{job_id}/renders/walkthrough.mp4")
    assert mp4.status_code == 200
    assert mp4.content[4:8] == b"ftyp"  # MP4 container magic
    assert len(mp4.content) > 10_000

    # Stage 12: the export bundle carries every deliverable.
    import io as _io
    import zipfile as _zipfile

    resp = client.get(f"/api/v1/jobs/{job_id}/export.zip")
    assert resp.status_code == 200
    names = set(_zipfile.ZipFile(_io.BytesIO(resp.content)).namelist())
    for expected in (
        "house.glb", "house.gltf", "house.obj", "house.ifc", "house.usdz",
        "renders/top_view.png", "renders/isometric.png", "renders/front.png",
        "renders/rear.png", "renders/walkthrough.mp4", "metadata.json",
        "artifacts/analysis.json", "artifacts/rooms.json",
    ):
        assert expected in names, expected
    meta = _zipfile.ZipFile(_io.BytesIO(resp.content)).read("metadata.json")
    import json as _json

    metadata = _json.loads(meta)
    assert metadata["formats_pending"]["fbx"]
    assert metadata["validation"]["passed"] is True


# Keep this test LAST in the file: it deliberately drains the shared
# per-process token bucket, so any request-making test after it gets 429s.
def test_rate_limit_kicks_in(client):
    # Bucket capacity is settings.rate_limit_per_minute (60); drain it.
    codes = [client.get("/api/v1/jobs/x").status_code for _ in range(70)]
    assert 429 in codes
