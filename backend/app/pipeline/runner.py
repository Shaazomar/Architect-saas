"""Pipeline orchestrator: bytes in, PipelineResult out.

Pure function of its inputs — no I/O besides CPU work — so it can run under
FastAPI's threadpool today and inside a Celery worker unchanged tomorrow.
"""
from __future__ import annotations

from .classify import classify_rooms
from .detect import detect_structure
from .graph import build_graph
from .ocr import NullOcr, OcrEngine
from .preprocess import preprocess
from .reconstruct import reconstruct
from .types import PipelineResult
from .validate import validate
from .vectorize import vectorize


def run_pipeline(
    image_bytes: bytes,
    meters_per_px: float | None = None,
    ocr: OcrEngine | None = None,
) -> PipelineResult:
    ocr = ocr or NullOcr()

    plan = preprocess(image_bytes)
    structure = detect_structure(plan)
    vector = vectorize(plan, structure)
    plan_graph = build_graph(plan, structure, vector)
    plan_graph = classify_rooms(plan_graph)

    ocr_result = ocr.read(plan.ink)
    scale = meters_per_px or ocr_result.meters_per_px

    recon = reconstruct(vector, meters_per_px=scale)
    validation = validate(recon, plan_graph)

    rooms = [
        {
            "id": r.id,
            "label": r.label,
            "area_m2": round(r.area_px * recon.meters_per_px**2, 2),
            "centroid_px": [round(c, 1) for c in r.polygon.centroid.coords[0]],
        }
        for r in plan_graph.rooms
    ]
    adjacency = [
        {"a": a, "b": b, "opening": bool(d.get("opening", False))}
        for a, b, d in plan_graph.graph.edges(data=True)
    ]

    return PipelineResult(
        glb=recon.scene_glb,
        rooms=rooms,
        adjacency=adjacency,
        validation=validation,
        stats=recon.stats,
    )
