"""Shared data contracts between pipeline stages.

Every stage consumes and returns these types only, so any stage can be swapped
(e.g. the morphological wall detector for a trained segmentation model)
without touching its neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


@dataclass
class PreprocessedPlan:
    """Output of the preprocessing stage."""

    ink: np.ndarray          # uint8 binary mask, 255 = drawn ink, 0 = paper
    width: int
    height: int


@dataclass
class DetectedStructure:
    """Output of the structure-detection stage (walls today; doors/windows/
    columns join here when the ML detectors land)."""

    wall_mask: np.ndarray            # uint8 binary mask, 255 = wall
    wall_thickness_px: float         # estimated dominant wall thickness


@dataclass
class Room:
    id: int
    polygon: Polygon                 # in pixel coordinates until reconstruction
    label: str = "room"
    area_px: float = 0.0
    label_confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class VectorPlan:
    """Vectorized 2D geometry of the plan (pixel coordinates)."""

    walls: MultiPolygon
    rooms: list[Room]
    wall_thickness_px: float


@dataclass
class PlanGraph:
    """Room-connectivity graph. Nodes are room ids; an edge means the rooms
    share a wall, with ``opening=True`` when free space connects them (a door
    or open passage)."""

    graph: nx.Graph
    rooms: list[Room]


@dataclass
class ReconstructionResult:
    scene_glb: bytes
    meters_per_px: float
    stats: dict[str, Any] = field(default_factory=dict)
    furniture: list[dict[str, Any]] = field(default_factory=list)
    # Stage 4 artifact: what geometry was generated, to which standards.
    geometry_manifest: dict[str, Any] = field(default_factory=dict)
    # Stage 6 artifact: PBR material assignment per scene node.
    materials_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    glb: bytes
    rooms: list[dict[str, Any]]
    adjacency: list[dict[str, Any]]
    validation: dict[str, Any]
    stats: dict[str, Any]
    furniture: list[dict[str, Any]] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    openings: list[dict[str, Any]] = field(default_factory=list)
    scene_graph: dict[str, Any] = field(default_factory=dict)
    # Stage 1 (Floor Plan Analyzer) artifact: pure detection, no geometry.
    analysis: dict[str, Any] = field(default_factory=dict)
    # Stage 2 (Room Detector) artifact: per-room semantics.
    rooms_detail: dict[str, Any] = field(default_factory=dict)
    # Stage 3 (Building Graph) artifact: semantic graph with zones/access.
    building_graph: dict[str, Any] = field(default_factory=dict)
    # Stage 4 (Geometry Engine) artifact: generated elements + standards.
    geometry: dict[str, Any] = field(default_factory=dict)
    # Stage 5 (Furniture AI) artifact: furnishing scene per room.
    furnishing: dict[str, Any] = field(default_factory=dict)
    # Stage 6 (Material AI) artifact: PBR materials + per-node assignments.
    materials: dict[str, Any] = field(default_factory=dict)
    # Stage 8 (Lighting AI) artifact: embedded lights + renderer settings.
    lighting: dict[str, Any] = field(default_factory=dict)
    # Stage 9 (Camera AI) artifact: embedded cameras + walkthrough path.
    cameras: dict[str, Any] = field(default_factory=dict)
