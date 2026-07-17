"""HTTP API.

POST /api/v1/plans                 upload a floor plan, returns a job id
GET  /api/v1/jobs/{id}             job status + analysis result
GET  /api/v1/jobs/{id}/model.glb   the reconstructed 3D model
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

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


def _process(job_id: str, image_bytes: bytes, meters_per_px: float | None) -> None:
    store.update_job(job_id, "processing")
    try:
        result = run_pipeline(image_bytes, meters_per_px=meters_per_px)
        job = store.get_job(job_id)
        assert job is not None
        (job.dir / "model.glb").write_bytes(result.glb)
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
            },
        )
    except PlanImageError as exc:
        store.update_job(job_id, "failed", error=str(exc))
    except Exception:
        # Never leak internals to the client; full trace goes to the log.
        log.exception("pipeline failed for job %s", job_id)
        store.update_job(job_id, "failed", error="Internal processing error.")


@router.post("/plans", status_code=202)
async def upload_plan(
    file: UploadFile,
    background: BackgroundTasks,
    meters_per_px: float | None = Query(default=None, gt=0, le=1.0),
):
    # Read one byte past the cap so we can distinguish "at limit" from "over".
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the upload size limit.")
    if _sniff(data[:16]) is None:
        raise HTTPException(415, "Only PNG or JPEG floor plan images are accepted.")

    job = store.create_job()
    (job.dir / "plan.png").write_bytes(data)
    background.add_task(_process, job.id, data, meters_per_px)
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
    "obj": "model/obj",
    "stl": "model/stl",
    "ply": "application/octet-stream",
}


def _convert_model(glb_path, fmt: str) -> bytes:
    import trimesh

    scene = trimesh.load(glb_path, file_type="glb")
    if fmt == "obj":
        data = scene.export(file_type="obj")
        return data.encode() if isinstance(data, str) else data
    # STL/PLY are single-mesh formats: bake transforms and concatenate.
    combined = trimesh.util.concatenate(scene.dump())
    data = combined.export(file_type=fmt)
    return data.encode() if isinstance(data, str) else data


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
