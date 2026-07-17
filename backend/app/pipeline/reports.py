"""Stage 8 — professional deliverable reports.

Produces the room schedule, area report, material quantity take-off and an
*indicative* cost estimate from the reconstructed geometry. Rates are
configuration values, not market data — the report says so explicitly.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from .types import PlanGraph, ReconstructionResult


def build_reports(plan_graph: PlanGraph, recon: ReconstructionResult) -> dict[str, Any]:
    mpp = recon.meters_per_px

    room_schedule = []
    for room in plan_graph.rooms:
        minx, miny, maxx, maxy = room.polygon.bounds
        room_schedule.append(
            {
                "id": room.id,
                "label": room.label,
                "area_m2": round(room.area_px * mpp**2, 2),
                "perimeter_m": round(room.polygon.exterior.length * mpp, 2),
                "approx_size_m": [round((maxx - minx) * mpp, 2), round((maxy - miny) * mpp, 2)],
                "connected_rooms": sorted(plan_graph.graph.neighbors(room.id))
                if room.id in plan_graph.graph
                else [],
            }
        )

    carpet_area = sum(r["area_m2"] for r in room_schedule)
    footprint_area = float(recon.stats.get("footprint_area_m2", carpet_area))
    wall_volume = float(recon.stats.get("wall_volume_m3", 0.0))
    # Paint on the interior face of each room's walls, floor to ceiling.
    paint_area = sum(r["perimeter_m"] for r in room_schedule) * settings.wall_height_m
    slab_volume = footprint_area * settings.slab_thickness_m
    roof_volume = footprint_area * settings.roof_thickness_m
    concrete_volume = wall_volume + slab_volume + roof_volume

    materials = {
        "carpet_area_m2": round(carpet_area, 1),
        "built_up_area_m2": round(footprint_area, 1),
        "wall_volume_m3": round(wall_volume, 1),
        "slab_volume_m3": round(slab_volume, 1),
        "roof_volume_m3": round(roof_volume, 1),
        "concrete_total_m3": round(concrete_volume, 1),
        "paint_area_m2": round(paint_area, 1),
        "flooring_area_m2": round(carpet_area, 1),
    }

    cost_items = {
        "structure_concrete": round(concrete_volume * settings.rate_concrete_per_m3),
        "flooring": round(carpet_area * settings.rate_flooring_per_m2),
        "painting": round(paint_area * settings.rate_paint_per_m2),
    }
    cost_estimate = {
        "currency": settings.currency,
        "items": cost_items,
        "total": sum(cost_items.values()),
        "disclaimer": "Indicative estimate from configured unit rates; not a quotation.",
    }

    return {
        "room_schedule": room_schedule,
        "materials": materials,
        "cost_estimate": cost_estimate,
    }
