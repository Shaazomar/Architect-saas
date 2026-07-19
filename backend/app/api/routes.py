"""HTTP API.

POST /api/v1/plans                 upload a floor plan, returns a job id
GET  /api/v1/jobs/{id}             job status + analysis result
GET  /api/v1/jobs/{id}/model.glb   the reconstructed 3D model
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from typing import Literal

from .. import store
from ..config import settings
from ..pipeline.preprocess import PlanImageError
from ..pipeline.runner import run_pipeline

log = logging.getLogger("architect")
router = APIRouter(prefix="/api/v1")

# Only raster plan images for now; PDF/DXF land later behind the same check.
_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
}


def _sniff(head: bytes) -> str | None:
    for magic, mime in _MAGIC.items():
        if head.startswith(magic):
            return mime
    return None


def _process(job_id: str, image_bytes: bytes, meters_per_px: float | None, furniture_mode: str) -> None:
    store.update_job(job_id, "processing")
    try:
        result = run_pipeline(image_bytes, meters_per_px=meters_per_px, furniture_mode=furniture_mode)
        job = store.get_job(job_id)
        assert job is not None
        (job.dir / "model.glb").write_bytes(result.glb)
        # Stage artifacts: each stage's output is its own deliverable.
        (job.dir / "analysis.json").write_text(json.dumps(result.analysis, indent=2))
        (job.dir / "rooms.json").write_text(json.dumps(result.rooms_detail, indent=2))
        (job.dir / "graph.json").write_text(json.dumps(result.building_graph, indent=2))
        (job.dir / "geometry.json").write_text(json.dumps(result.geometry, indent=2))
        (job.dir / "scene.json").write_text(json.dumps(result.furnishing, indent=2))
        (job.dir / "materials.json").write_text(json.dumps(result.materials, indent=2))
        (job.dir / "lighting.json").write_text(json.dumps(result.lighting, indent=2))
        (job.dir / "cameras.json").write_text(json.dumps(result.cameras, indent=2))
        renders = _render_views(job)
        store.update_job(
            job_id,
            "done",
            result={
                "rooms": result.rooms,
                "adjacency": result.adjacency,
                "validation": result.validation,
                "stats": result.stats,
                "furniture": result.furniture,
                "reports": result.reports,
                "openings": result.openings,
                "scene_graph": result.scene_graph,
                "renders": renders,
            },
        )
    except PlanImageError as exc:
        store.update_job(job_id, "failed", error=str(exc))
    except Exception:
        # Never leak internals to the client; full trace goes to the log.
        log.exception("pipeline failed for job %s", job_id)
        store.update_job(job_id, "failed", error="Internal processing error.")


def _render_views(job) -> dict:
    """Stage 10: render the camera set to PNGs in a subprocess (GL contexts
    must own a process main thread on macOS). Failure is graceful: the job
    still completes and renders.json records the reason."""
    if not settings.renders_enabled:
        skipped = {"stage": "renders", "status": "skipped", "reason": "renders disabled by configuration"}
        (job.dir / "renders.json").write_text(json.dumps(skipped, indent=2))
        return skipped
    try:
        subprocess.run(
            [
                sys.executable, "-m", "app.pipeline.renders",
                str(job.dir / "model.glb"), str(job.dir / "cameras.json"),
                str(job.dir / "renders"),
                str(settings.render_width), str(settings.render_height),
            ],
            capture_output=True,
            timeout=settings.render_timeout_s,
            check=False,
        )
        manifest_path = job.dir / "renders.json"
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text())
    except Exception:
        log.exception("render subprocess failed for job %s", job.id)
    fallback = {"stage": "renders", "status": "failed", "reason": "renderer unavailable on this host"}
    (job.dir / "renders.json").write_text(json.dumps(fallback, indent=2))
    return fallback


@router.post("/plans", status_code=202)
async def upload_plan(
    file: UploadFile,
    background: BackgroundTasks,
    meters_per_px: float | None = Query(default=None, gt=0, le=1.0),
    furniture: Literal["auto", "detected", "generated", "none"] = Query(default="auto"),
):
    # Read one byte past the cap so we can distinguish "at limit" from "over".
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the upload size limit.")
    if _sniff(data[:16]) is None:
        raise HTTPException(415, "Only PNG or JPEG floor plan images are accepted.")

    job = store.create_job()
    (job.dir / "plan.png").write_bytes(data)
    background.add_task(_process, job.id, data, meters_per_px, furniture)
    return {"job_id": job.id, "status": job.status}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return {"job_id": job.id, "status": job.status, "error": job.error, "result": job.result}


# Format whitelist: anything else in the URL is a 404, so no user-controlled
# value ever reaches the filesystem or an exporter.
_EXPORT_FORMATS = {
    "glb": "model/gltf-binary",
    "gltf": "model/gltf+json",
    "obj": "model/obj",
    "stl": "model/stl",
    "ply": "application/octet-stream",
    "ifc": "application/x-step",
    "usdz": "model/vnd.usdz+zip",
}


def _convert_model(glb_path, fmt: str) -> bytes:
    import trimesh

    from ..pipeline import exports

    if fmt in ("gltf", "ifc", "usdz"):
        glb = glb_path.read_bytes()
        if fmt == "gltf":
            return exports.glb_to_gltf_embedded(glb)
        if fmt == "ifc":
            return exports.export_ifc(glb)
        return exports.export_usdz(glb)

    scene = trimesh.load(glb_path, file_type="glb")
    if fmt == "obj":
        data = scene.export(file_type="obj")
        return data.encode() if isinstance(data, str) else data
    # STL/PLY are single-mesh formats: bake transforms and concatenate.
    combined = trimesh.util.concatenate(scene.dump())
    data = combined.export(file_type=fmt)
    return data.encode() if isinstance(data, str) else data


# Whitelisted per-stage artifacts. User input never reaches the filesystem:
# the name must be an exact key here and the job id must exist in the store.
_STAGE_ARTIFACTS = {
    "analysis.json", "rooms.json", "graph.json", "geometry.json",
    "scene.json", "materials.json", "lighting.json", "cameras.json", "renders.json",
}
_RENDER_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


# Friendly deliverable names for the export bundle (Stage 12 spec).
_FRIENDLY_VIEWS = {
    "top_orthographic": "top_view",
    "isometric_45": "isometric",
    "bird_eye": "bird_eye",
    "front_elevation": "front",
    "rear_elevation": "rear",
}


def _friendly_view_name(stem: str) -> str:
    if stem in _FRIENDLY_VIEWS:
        return _FRIENDLY_VIEWS[stem]
    if stem.startswith("interior_"):  # interior_{room_id}_{label}
        return stem.split("_", 2)[2]
    return stem


@router.get("/jobs/{job_id}/renders/walkthrough.mp4")
def job_walkthrough(job_id: str):
    job = store.get_job(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Walkthrough not available.")
    path = job.dir / "renders" / "walkthrough.mp4"
    if not path.is_file():
        raise HTTPException(404, "Walkthrough not available.")
    return FileResponse(path, media_type="video/mp4", filename="walkthrough.mp4")


@router.get("/jobs/{job_id}/export.zip")
def job_export_bundle(job_id: str):
    """Stage 12: one bundle with every deliverable — models in all formats,
    renders under friendly names, the walkthrough, metadata and all stage
    artifacts. Built once, then cached."""
    import zipfile

    job = store.get_job(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Export not available.")
    bundle = job.dir / "export.zip"
    if not bundle.is_file():
        glb_path = job.dir / "model.glb"
        if not glb_path.is_file():
            raise HTTPException(404, "Export not available.")
        skipped: dict[str, str] = {}
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            for fmt in _EXPORT_FORMATS:
                path = job.dir / f"model.{fmt}"
                try:
                    if fmt != "glb" and not path.is_file():
                        path.write_bytes(_convert_model(glb_path, fmt))
                    z.write(path, f"house.{fmt}")
                except Exception as exc:
                    log.warning("export %s failed for job %s: %s", fmt, job.id, exc)
                    skipped[fmt] = f"{type(exc).__name__}: {exc}"
            renders_dir = job.dir / "renders"
            if renders_dir.is_dir():
                for png in sorted(renders_dir.glob("*.png")):
                    z.write(png, f"renders/{_friendly_view_name(png.stem)}.png")
                mp4 = renders_dir / "walkthrough.mp4"
                if mp4.is_file():
                    z.write(mp4, "renders/walkthrough.mp4")
            for artifact in sorted(_STAGE_ARTIFACTS):
                path = job.dir / artifact
                if path.is_file():
                    z.write(path, f"artifacts/{artifact}")
            metadata = dict(job.result or {})
            metadata.update(
                {
                    "job_id": job.id,
                    "formats_skipped": skipped or None,
                    "formats_pending": {"fbx": "proprietary Autodesk SDK format; use the IFC or glTF"},
                    "optimization": {
                        "vertex_welding": "applied to IFC/USDZ exports",
                        "draco_meshopt_compression": "pending",
                        "textures": "factor-based PBR (no texture maps yet); nothing to embed",
                    },
                }
            )
            z.writestr("metadata.json", json.dumps(metadata, indent=2))
    return FileResponse(bundle, media_type="application/zip", filename="export.zip")


@router.get("/jobs/{job_id}/renders/{name}.png")
def job_render(job_id: str, name: str):
    if not _RENDER_NAME.match(name):
        raise HTTPException(404, "Render not found.")
    job = store.get_job(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Render not found.")
    path = job.dir / "renders" / f"{name}.png"
    if not path.is_file():
        raise HTTPException(404, "Render not found.")
    return FileResponse(path, media_type="image/png", filename=f"{name}.png")


@router.get("/jobs/{job_id}/{artifact}.json")
def job_stage_artifact(job_id: str, artifact: str):
    filename = f"{artifact}.json"
    if filename not in _STAGE_ARTIFACTS:
        raise HTTPException(404, "Unknown artifact.")
    job = store.get_job(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Artifact not available.")
    path = job.dir / filename
    if not path.is_file():
        raise HTTPException(404, "Artifact not available.")
    return FileResponse(path, media_type="application/json", filename=filename)


@router.get("/jobs/{job_id}/model.{fmt}")
def job_model(job_id: str, fmt: str):
    media_type = _EXPORT_FORMATS.get(fmt)
    if media_type is None:
        raise HTTPException(404, "Unsupported export format.")
    job = store.get_job(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "Model not available.")
    glb_path = job.dir / "model.glb"
    if not glb_path.is_file():
        raise HTTPException(404, "Model not available.")

    path = job.dir / f"model.{fmt}"
    if fmt != "glb" and not path.is_file():
        path.write_bytes(_convert_model(glb_path, fmt))
    return FileResponse(path, media_type=media_type, filename=f"model.{fmt}")
