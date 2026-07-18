"""Pipeline orchestrator: bytes in, PipelineResult out.

Pure function of its inputs — no I/O besides CPU work — so it can run under
FastAPI's threadpool today and inside a Celery worker unchanged tomorrow.

furniture_mode:
  "detected"  (default) reconstruct only furniture symbols found in the
              drawing, at their exact drawn position — nothing is invented;
  "generated" opt-in constraint-based catalog placement for empty plans;
  "none"      structure only.
"""
from __future__ import annotations

from .building_graph import build_semantic_graph
from .classify import classify_rooms
from .detect import detect_structure
from .graph import build_graph
from .ocr import NullOcr, OcrEngine
from .openings import detect_openings
from .preprocess import preprocess
from .reconstruct import reconstruct
from .reports import build_reports
from .symbols import detect_symbols
from .types import PipelineResult
from .validate import validate
from .vectorize import vectorize

FURNITURE_MODES = ("detected", "generated", "none")


def run_pipeline(
    image_bytes: bytes,
    meters_per_px: float | None = None,
    furniture_mode: str = "detected",
    ocr: OcrEngine | None = None,
) -> PipelineResult:
    if furniture_mode not in FURNITURE_MODES:
        raise ValueError(f"furniture_mode must be one of {FURNITURE_MODES}")
    ocr = ocr or NullOcr()

    plan = preprocess(image_bytes)
    structure = detect_structure(plan)
    vector = vectorize(plan, structure)
    plan_graph = build_graph(plan, structure, vector)

    ocr_result = ocr.read(plan.ink)
    scale = meters_per_px or ocr_result.meters_per_px
    scale_source = "explicit" if meters_per_px else (
        "ocr" if ocr_result.meters_per_px else "wall_thickness_assumption"
    )

    # Effective scale is needed by the classifier/detectors even when inferred.
    from ..config import settings

    effective_scale = scale or (settings.wall_thickness_m / vector.wall_thickness_px)

    plan_graph = classify_rooms(plan_graph, effective_scale)

    openings = detect_openings(plan, structure, vector, effective_scale)
    detected = detect_symbols(plan, structure, vector, effective_scale)

    recon = reconstruct(
        vector,
        meters_per_px=scale,
        furniture_mode=furniture_mode,
        detected_objects=detected,
    )
    validation = validate(recon, plan_graph)

    rooms = [
        {
            "id": r.id,
            "label": r.label,
            "label_confidence": r.label_confidence,
            "label_source": "heuristic",       # OCR/GNN stages will override
            "evidence": r.evidence,
            "area_m2": round(r.area_px * recon.meters_per_px**2, 2),
            "centroid_px": [round(c, 1) for c in r.polygon.centroid.coords[0]],
        }
        for r in plan_graph.rooms
    ]
    adjacency = [
        {"a": a, "b": b, "opening": bool(d.get("opening", False))}
        for a, b, d in plan_graph.graph.edges(data=True)
    ]
    openings_out = [
        {
            "id": o.id,
            "kind": o.kind,
            "width_m": o.width_m,
            "rooms": o.rooms,
            "center_px": list(o.center_px),
            "swing_direction": None,  # needs the door-leaf symbol (ML stage)
            "confidence": o.confidence,
        }
        for o in openings
    ]

    # Stage 3 artifact — Semantic Building Graph.
    semantic_graph = build_semantic_graph(plan_graph, openings_out, effective_scale)

    # Stage 2 artifact — Room Detector: every room with polygon, center, name,
    # area, doors, windows (pending), adjacency and confidence + evidence.
    rooms_detail = {
        "stage": "room_detection",
        "classifier": "heuristic_rules",
        "rooms": [
            {
                "id": r.id,
                "name": r.label,
                "confidence": r.label_confidence,
                "evidence": r.evidence,
                "polygon_px": [[round(x, 1), round(y, 1)] for x, y in r.polygon.exterior.coords],
                "center_px": [round(c, 1) for c in r.polygon.centroid.coords[0]],
                "center_m": [
                    round(r.polygon.centroid.x * effective_scale, 2),
                    round(r.polygon.centroid.y * effective_scale, 2),
                ],
                "area_m2": round(r.area_px * effective_scale**2, 2),
                "perimeter_m": round(r.polygon.exterior.length * effective_scale, 2),
                "doors": [o["id"] for o in openings_out if r.id in o["rooms"]],
                "windows": {"items": [], "status": "pending_ml_detector"},
                "adjacent_rooms": sorted(plan_graph.graph.neighbors(r.id))
                if r.id in plan_graph.graph
                else [],
            }
            for r in plan_graph.rooms
        ],
    }

    # Stage 1 artifact — Floor Plan Analyzer: detection only, no geometry.
    # Element classes we cannot detect yet are empty lists, never guesses;
    # each carries a status naming the stage that will fill it.
    analysis = {
        "stage": "floor_plan_analysis",
        "detector": "classical_cv",
        "image_px": {"width": plan.width, "height": plan.height},
        "scale": {
            "meters_per_px": round(effective_scale, 6),
            "source": scale_source,
            "confidence": 0.9 if scale_source == "explicit" else 0.5,
        },
        "walls": [
            {
                "id": i,
                "polygon_px": [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords],
                "hole_count": len(g.interiors),
            }
            for i, g in enumerate(vector.walls.geoms)
        ],
        "wall_thickness_px": round(vector.wall_thickness_px, 1),
        "rooms": [
            {
                "id": r.id,
                "label": r.label,
                "label_source": "heuristic",
                "polygon_px": [[round(x, 1), round(y, 1)] for x, y in r.polygon.exterior.coords],
            }
            for r in vector.rooms
        ],
        "doors": openings_out,
        "windows": {"items": [], "status": "pending_ml_detector"},
        "stairs": {"items": [], "status": "pending_ml_detector"},
        "columns": {"items": [], "status": "pending_ml_detector"},
        "dimensions": {"items": [], "status": "pending_ocr"},
        "text": {"items": [], "status": "pending_ocr"},
        "north_arrow": {"detected": False, "status": "pending_ml_detector"},
        "symbols": [
            {
                "id": s.id,
                "category": s.category,
                "confidence": s.confidence,
                "room_id": s.room_id,
                "bbox_px": [round(v, 1) for v in s.footprint_px.bounds],
                "size_m": list(s.size_m),
                "rotation_deg": s.rotation_deg,
            }
            for s in detected
        ],
    }

    scene_graph = {
        "building": {
            "rooms": [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "objects": [
                        f["item"] + f"_{i}"
                        for i, f in enumerate(recon.furniture)
                        if f["room_id"] == r["id"]
                    ],
                    "openings": [o["id"] for o in openings_out if r["id"] in o["rooms"]],
                }
                for r in rooms
            ]
        }
    }

    validation.update(
        {
            "wall_count": recon.stats.get("wall_mesh_count"),
            "opening_count": len(openings_out),
            "door_count": sum(1 for o in openings_out if o["kind"] == "door"),
            "window_count": 0,  # window symbols need the ML detector stage
            "furniture_count": len(recon.furniture),
            "scale_source": scale_source,
            "scale_confidence": 0.9 if scale_source == "explicit" else 0.5,
            "warnings": [
                w
                for w in (
                    "Scale inferred from assumed 230 mm exterior wall thickness."
                    if scale_source == "wall_thickness_assumption"
                    else None,
                    "Room labels are heuristic (adjacency/area); OCR and GNN stages pending."
                    if rooms
                    else None,
                    "Window and door-swing symbols are not detected yet (ML stage pending)."
                )
                if w
            ],
        }
    )

    return PipelineResult(
        glb=recon.scene_glb,
        rooms=rooms,
        adjacency=adjacency,
        validation=validation,
        stats=recon.stats,
        furniture=recon.furniture,
        reports=build_reports(plan_graph, recon),
        openings=openings_out,
        scene_graph=scene_graph,
        analysis=analysis,
        rooms_detail=rooms_detail,
        building_graph=semantic_graph,
    )
