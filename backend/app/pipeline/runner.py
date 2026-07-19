"""Pipeline orchestrator: bytes in, PipelineResult out.

Stage order matters:
detection (1-3) -> geometry + furnishing (4-7) -> validation & repair (11,
on the plain GLB so repairs re-export cleanly) -> light injection (8) ->
camera injection (9) -> post-injection checks. Rendering (10) is I/O- and
GPU-bound, so it runs in the API layer as a subprocess, not here — this
function stays a pure function of its inputs and can move into a Celery
worker unchanged.

furniture_mode:
  "auto"      (default) rooms keep their drawn symbols at exact position;
              rooms with no symbols are furnished so no room is left empty;
  "detected"  strictly what the drawing shows;
  "generated" catalog placement everywhere;
  "none"      structure only.
"""
from __future__ import annotations

from ..config import settings
from .building_graph import build_semantic_graph
from .cameras import build_cameras, build_walkthrough, inject_cameras
from .classify import classify_rooms
from .detect import detect_structure
from .graph import build_graph
from .lighting import build_lights, inject_lights, lighting_manifest
from .ocr import NullOcr, OcrEngine
from .openings import detect_openings
from .preprocess import preprocess
from .reconstruct import reconstruct
from .reports import build_reports
from .symbols import detect_symbols
from .types import PipelineResult
from .validate import validate_and_repair
from .vectorize import vectorize

FURNITURE_MODES = ("auto", "detected", "generated", "none")


def run_pipeline(
    image_bytes: bytes,
    meters_per_px: float | None = None,
    furniture_mode: str = "auto",
    ocr: OcrEngine | None = None,
) -> PipelineResult:
    if furniture_mode not in FURNITURE_MODES:
        raise ValueError(f"furniture_mode must be one of {FURNITURE_MODES}")
    ocr = ocr or NullOcr()

    # ---- Stages 1-3: detection and semantics ----
    plan = preprocess(image_bytes)
    structure = detect_structure(plan)
    vector = vectorize(plan, structure)
    plan_graph = build_graph(plan, structure, vector)

    ocr_result = ocr.read(plan.ink)
    scale = meters_per_px or ocr_result.meters_per_px
    scale_source = "explicit" if meters_per_px else (
        "ocr" if ocr_result.meters_per_px else "wall_thickness_assumption"
    )
    effective_scale = scale or (settings.wall_thickness_m / vector.wall_thickness_px)

    plan_graph = classify_rooms(plan_graph, effective_scale)
    openings = detect_openings(plan, structure, vector, effective_scale)
    detected = detect_symbols(plan, structure, vector, effective_scale)

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
    semantic_graph = build_semantic_graph(plan_graph, openings_out, effective_scale)

    # ---- Stages 4-7: geometry, furnishing, materials ----
    recon = reconstruct(
        vector,
        meters_per_px=scale,
        furniture_mode=furniture_mode,
        detected_objects=detected,
        openings=openings,
    )

    # ---- Stage 11: validate and auto-repair on the plain GLB ----
    validation, repaired_glb = validate_and_repair(
        recon,
        plan_graph,
        assignments=recon.materials_manifest.get("assignments"),
        furniture_sources=recon.geometry_manifest.get("furniture_nodes"),
    )
    if repaired_glb is not None:
        recon.scene_glb = repaired_glb
        # Drop repaired (removed) pieces from the furniture report so the
        # report matches the shipped scene. Node: furniture_{room}_{item}_{idx}
        gone_prefixes = {n.rsplit("_", 1)[0] for n in validation.get("repaired_nodes", [])}
        recon.furniture = [
            f for f in recon.furniture
            if f["source"] != "generated"
            or f"furniture_{f['room_id']}_{f['item']}" not in gone_prefixes
        ]

    # ---- Stage 8: embed punctual lights ----
    lights = build_lights(plan_graph.rooms, effective_scale, settings.wall_height_m)
    recon.scene_glb = inject_lights(recon.scene_glb, lights)

    # ---- Stage 9: embed cameras ----
    cameras = build_cameras(plan_graph.rooms, effective_scale, settings.wall_height_m,
                            semantic_graph["accessibility"])
    walkthrough = build_walkthrough(plan_graph.rooms, effective_scale,
                                    semantic_graph["accessibility"])
    recon.scene_glb = inject_cameras(recon.scene_glb, cameras)
    cameras_doc = {"stage": "cameras", "cameras": cameras, "walkthrough": walkthrough}

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

    # Stage 2 artifact — Room Detector.
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
            for r in plan_graph.rooms
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
            "lights_embedded": len(lights),
            "cameras_embedded": len(cameras),
            "hierarchy_consistent": len(semantic_graph["nodes"]) == len(rooms),
            "accessibility_ok": semantic_graph["accessibility"]["all_rooms_reachable"],
            "warnings": [
                w
                for w in (
                    "Scale inferred from assumed 230 mm exterior wall thickness."
                    if scale_source == "wall_thickness_assumption"
                    else None,
                    "Room labels are heuristic (adjacency/area); OCR and GNN stages pending."
                    if rooms
                    else None,
                    "Window and door-swing symbols are not detected yet (ML stage pending).",
                )
                if w
            ],
        }
    )

    # Stage 5+7 artifact — furnishing + decor scene.
    furnishing = {
        "stage": "furnishing",
        "mode": furniture_mode,
        "rules": {
            "circulation_clearance_m": 0.45,
            "door_clearance": "no generated furniture within door half-width + 0.6 m",
            "policy": "detected symbols keep exact drawn position; unplaceable items are skipped",
        },
        "rooms": [
            {
                "id": r.id,
                "label": r.label,
                "items": [f for f in recon.furniture if f["room_id"] == r.id],
            }
            for r in plan_graph.rooms
        ],
        "totals": {
            "items": len(recon.furniture),
            "detected": sum(1 for f in recon.furniture if f["source"] == "detected"),
            "generated": sum(1 for f in recon.furniture if f["source"] == "generated"),
        },
    }

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
        geometry=recon.geometry_manifest,
        furnishing=furnishing,
        materials=recon.materials_manifest,
        lighting=lighting_manifest(lights),
        cameras=cameras_doc,
    )
