"""Stage 8 — Lighting AI.

Two deliverables:
1. Physical lights embedded in the GLB via the standard KHR_lights_punctual
   extension (trimesh does not serialize lights, so we inject them into the
   glTF JSON chunk directly): one warm ceiling light per room (cool-white
   task lighting in kitchens/wet areas, intensity scaled by room area) and a
   directional sun. Three.js / Babylon / Blender all consume this natively,
   so the model arrives lit.
2. A lighting.json artifact recording every light plus renderer
   recommendations (HDRI, exposure, soft shadows, AO) — settings that live
   in the renderer, not the file, and are therefore documented rather than
   pretended.

Frame note: a point at image-pixel (x, y) at height h lands at glTF position
(x*scale, h, y*scale) after reconstruction's mirror + rotation.
"""
from __future__ import annotations

import json
import math
import struct
from typing import Any

_WARM = [1.0, 0.86, 0.72]      # ~2700 K living areas
_COOL = [1.0, 0.96, 0.9]       # ~4000 K task lighting
_TASK_ROOMS = {"kitchen", "bathroom", "common_toilet", "utility", "garage", "store"}


def _sun_quaternion_xyzw(pitch_deg: float, yaw_deg: float) -> list[float]:
    """Quaternion for Ry(yaw) . Rx(pitch); a directional light shines -Z."""
    sp, cp = math.sin(math.radians(pitch_deg) / 2), math.cos(math.radians(pitch_deg) / 2)
    sy, cy = math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)
    return [cy * sp, cp * sy, -sy * sp, cy * cp]


def build_lights(rooms, meters_per_px: float, ceiling_h: float) -> list[dict[str, Any]]:
    lights: list[dict[str, Any]] = [
        {
            "name": "sun",
            "type": "directional",
            "color": [1.0, 0.98, 0.9],
            "intensity": 4.0,
            "rotation_quat_xyzw": _sun_quaternion_xyzw(-50.0, 35.0),
            "purpose": "sun / natural light through openings",
        }
    ]
    for r in rooms:
        c = r.polygon.centroid
        area_m2 = r.area_px * meters_per_px**2
        task = r.label in _TASK_ROOMS
        lights.append(
            {
                "name": f"ceiling_{r.id}_{r.label}",
                "type": "point",
                "color": _COOL if task else _WARM,
                "intensity": round(min(90.0, max(20.0, area_m2 * 0.6)), 1),
                "position": [
                    round(c.x * meters_per_px, 2),
                    round(ceiling_h - 0.15, 2),
                    round(c.y * meters_per_px, 2),
                ],
                "room_id": r.id,
                "purpose": "task ceiling light" if task else "warm ceiling light",
            }
        )
    return lights


def inject_lights(glb: bytes, lights: list[dict[str, Any]]) -> bytes:
    """Add KHR_lights_punctual lights + light nodes into a GLB's JSON chunk."""
    if glb[:4] != b"glTF" or not lights:
        return glb
    json_len = struct.unpack("<I", glb[12:16])[0]
    if glb[16:20] != b"JSON":
        return glb
    doc = json.loads(glb[20 : 20 + json_len])
    tail = glb[20 + json_len :]  # remaining chunks (BIN), byte-for-byte

    khr: list[dict[str, Any]] = []
    nodes = doc.setdefault("nodes", [])
    scene_nodes = doc["scenes"][doc.get("scene", 0)].setdefault("nodes", [])
    for light in lights:
        khr.append(
            {
                "name": light["name"],
                "type": light["type"],
                "color": light["color"],
                "intensity": light["intensity"],
            }
        )
        node: dict[str, Any] = {
            "name": f"light_{light['name']}",
            "extensions": {"KHR_lights_punctual": {"light": len(khr) - 1}},
        }
        if "position" in light:
            node["translation"] = [float(v) for v in light["position"]]
        if "rotation_quat_xyzw" in light:
            node["rotation"] = [float(v) for v in light["rotation_quat_xyzw"]]
        nodes.append(node)
        scene_nodes.append(len(nodes) - 1)

    used = doc.setdefault("extensionsUsed", [])
    if "KHR_lights_punctual" not in used:
        used.append("KHR_lights_punctual")
    doc.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": khr}

    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload) + len(tail)
    return (
        glb[:8]
        + struct.pack("<I", total)
        + struct.pack("<I", len(payload))
        + b"JSON"
        + payload
        + tail
    )


def lighting_manifest(lights: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": "lighting",
        "model": "KHR_lights_punctual (embedded in the GLB)",
        "lights": lights,
        "renderer_recommendations": {
            "environment": "neutral outdoor HDRI, ~1.0 EV",
            "exposure": 1.0,
            "tone_mapping": "ACES filmic",
            "shadows": "soft (PCF/PCSS), sun as key light",
            "ambient_occlusion": "SSAO/GTAO on",
            "note": "these are renderer settings, not file data; the GLB carries the lights themselves",
        },
    }
