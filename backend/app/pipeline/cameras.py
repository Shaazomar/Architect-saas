"""Stage 9 — Camera AI.

Generates a full presentation camera set — orthographic top, 45° isometric,
bird's eye, front/rear elevations, an eye-level interior camera per room,
and an interior walkthrough path ordered by door-depth from the entrance —
and embeds the cameras into the GLB as standard glTF cameras (same JSON-chunk
injection approach as the lights), so any glTF viewer can jump straight to
each view.

Frame reminder: image-pixel (x, y) at height h sits at glTF (x*s, h, y*s);
+Z is the drawing's "down" (south, the front of the plan), the camera looks
along its local -Z with +Y up.
"""
from __future__ import annotations

import json
import math
import struct
from typing import Any

import numpy as np

_EYE_LEVEL = 1.55


def _look_at_quat_xyzw(eye, target, up=(0.0, 1.0, 0.0)) -> list[float]:
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    n = np.linalg.norm(f)
    f = f / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    if abs(np.dot(f, up)) > 0.999:  # looking straight up/down: pick a stable up
        up = np.array([0.0, 0.0, -1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, f)
    # Camera looks along -Z: rotation matrix columns are (right, up, -forward).
    m = np.column_stack([right, true_up, -f])
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _cam(name: str, kind: str, eye, target, *, ortho_mag: float | None = None,
         yfov: float = 0.9, hide_roof: bool = False, room_id: int | None = None,
         purpose: str = "") -> dict[str, Any]:
    cam: dict[str, Any] = {
        "name": name,
        "projection": "orthographic" if ortho_mag is not None else "perspective",
        "position": [round(float(v), 2) for v in eye],
        "target": [round(float(v), 2) for v in target],
        "rotation_quat_xyzw": _look_at_quat_xyzw(eye, target),
        "hide_roof": hide_roof,
        "purpose": purpose or kind,
    }
    if ortho_mag is not None:
        cam["ortho_mag"] = round(ortho_mag, 2)
    else:
        cam["yfov"] = yfov
    if room_id is not None:
        cam["room_id"] = room_id
    return cam


def build_cameras(rooms, meters_per_px: float, wall_h: float, accessibility: dict) -> list[dict[str, Any]]:
    s = meters_per_px
    xs = [x * s for r in rooms for x, _ in r.polygon.exterior.coords]
    zs = [y * s for r in rooms for _, y in r.polygon.exterior.coords]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    dx, dz = max(xs) - min(xs), max(zs) - min(zs)
    d = max(dx, dz)
    center = (cx, wall_h / 2, cz)
    ground = (cx, 0.0, cz)

    cams = [
        _cam("top_orthographic", "top", (cx, d * 1.4, cz + 0.01), ground,
             ortho_mag=d * 0.62, hide_roof=True, purpose="orthographic top plan view"),
        _cam("isometric_45", "isometric", (cx + d * 0.85, d * 0.85, cz + d * 0.85), center,
             hide_roof=True, purpose="45-degree isometric overview"),
        _cam("bird_eye", "bird_eye", (cx + d * 0.35, d * 1.1, cz + d * 0.55), ground,
             hide_roof=True, purpose="bird's-eye view"),
        _cam("front_elevation", "front", (cx, wall_h * 0.55, max(zs) + d * 1.2), center,
             ortho_mag=max(dx, wall_h) * 0.62, purpose="front elevation (drawing's south)"),
        _cam("rear_elevation", "rear", (cx, wall_h * 0.55, min(zs) - d * 1.2), center,
             ortho_mag=max(dx, wall_h) * 0.62, purpose="rear elevation"),
    ]

    for r in rooms:
        c = r.polygon.centroid
        eye = (c.x * s, _EYE_LEVEL, c.y * s)
        # Aim at the farthest boundary point for the widest interior sweep.
        far = max(r.polygon.exterior.coords,
                  key=lambda p: (p[0] - c.x) ** 2 + (p[1] - c.y) ** 2)
        target = ((c.x + (far[0] - c.x) * 0.8) * s, 1.2, (c.y + (far[1] - c.y) * 0.8) * s)
        cams.append(
            _cam(f"interior_{r.id}_{r.label}", "interior", eye, target,
                 yfov=1.1, hide_roof=True, room_id=r.id,
                 purpose=f"eye-level interior view of the {r.label.replace('_', ' ')}")
        )
    return cams


def build_walkthrough(rooms, meters_per_px: float, accessibility: dict) -> dict[str, Any]:
    """Ordered eye-level waypoints: entrance first, then rooms by door depth."""
    depth = {int(k): v for k, v in accessibility.get("door_depth_from_entrance", {}).items()}
    ordered = sorted(rooms, key=lambda r: (depth.get(r.id, 99), r.id))
    s = meters_per_px
    waypoints = [
        {
            "room_id": r.id,
            "label": r.label,
            "position": [round(r.polygon.centroid.x * s, 2), _EYE_LEVEL,
                         round(r.polygon.centroid.y * s, 2)],
            "door_depth": depth.get(r.id),
        }
        for r in ordered
    ]
    return {
        "kind": "interior_walkthrough",
        "waypoints": waypoints,
        "animation": "pending — glTF animation export planned; waypoints are renderer-ready",
    }


def inject_cameras(glb: bytes, cameras: list[dict[str, Any]], znear: float = 0.05,
                   zfar: float = 500.0) -> bytes:
    """Embed the cameras into the GLB as standard glTF cameras + nodes."""
    if glb[:4] != b"glTF" or not cameras:
        return glb
    json_len = struct.unpack("<I", glb[12:16])[0]
    if glb[16:20] != b"JSON":
        return glb
    doc = json.loads(glb[20 : 20 + json_len])
    tail = glb[20 + json_len :]

    gltf_cams = doc.setdefault("cameras", [])
    nodes = doc.setdefault("nodes", [])
    scene_nodes = doc["scenes"][doc.get("scene", 0)].setdefault("nodes", [])
    for cam in cameras:
        if cam["projection"] == "orthographic":
            entry = {
                "name": cam["name"],
                "type": "orthographic",
                "orthographic": {"xmag": cam["ortho_mag"], "ymag": cam["ortho_mag"],
                                 "znear": znear, "zfar": zfar},
            }
        else:
            entry = {
                "name": cam["name"],
                "type": "perspective",
                "perspective": {"yfov": cam["yfov"], "znear": znear},
            }
        gltf_cams.append(entry)
        nodes.append(
            {
                "name": f"camera_{cam['name']}",
                "camera": len(gltf_cams) - 1,
                "translation": [float(v) for v in cam["position"]],
                "rotation": [float(v) for v in cam["rotation_quat_xyzw"]],
            }
        )
        scene_nodes.append(len(nodes) - 1)

    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload) + len(tail)
    return glb[:8] + struct.pack("<I", total) + struct.pack("<I", len(payload)) + b"JSON" + payload + tail
