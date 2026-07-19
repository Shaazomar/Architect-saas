"""Stage 10 — Render AI (rasterized presentation renders).

Renders every Stage 9 camera to PNG using pyrender (GPU rasterization with
the PBR materials and the embedded punctual lights). This yields real,
z-buffered, lit architectural views — top plan, isometric, elevations and
eye-level interiors. Path-traced photorealism (Cycles-grade GI, soft area
shadows) is a renderer capability this host does not have; the manifest
declares it pending rather than passing rasterized output off as
photorealistic.

Runs as a subprocess (``python -m app.pipeline.renders``) because macOS
requires GL contexts on a process main thread — the API's worker threads
must never touch GL.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


def _quat_to_mat(q_xyzw, translation) -> np.ndarray:
    x, y, z, w = q_xyzw
    m = np.eye(4)
    m[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m[:3, 3] = translation
    return m


def _glb_json(glb: bytes) -> dict:
    jlen = struct.unpack("<I", glb[12:16])[0]
    return json.loads(glb[20 : 20 + jlen])


def render_all(glb_path: str, cameras_path: str, out_dir: str, width: int, height: int) -> dict:
    import pyrender
    import trimesh
    from PIL import Image

    glb = Path(glb_path).read_bytes()
    tm_scene = trimesh.load(glb_path, file_type="glb")
    cameras_doc = json.loads(Path(cameras_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Lights from the GLB's KHR_lights_punctual extension.
    doc = _glb_json(glb)
    lights = []
    khr = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    for node in doc.get("nodes", []):
        ref = node.get("extensions", {}).get("KHR_lights_punctual", {}).get("light")
        if ref is None:
            continue
        light = khr[ref]
        lights.append(
            {
                "type": light["type"],
                "color": light["color"],
                "intensity": light["intensity"],
                "translation": node.get("translation", [0, 0, 0]),
                "rotation": node.get("rotation", [0, 0, 0, 1]),
            }
        )

    renderer = pyrender.OffscreenRenderer(width, height)
    views = []
    for cam in cameras_doc["cameras"]:
        scene = pyrender.Scene(ambient_light=[0.32, 0.32, 0.34], bg_color=[0.96, 0.96, 0.97])
        for name, geom in tm_scene.geometry.items():
            if cam.get("hide_roof") and name.startswith(("roof_", "roof_parapet")):
                continue
            if not isinstance(geom, trimesh.Trimesh):
                continue
            scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False), name=name)

        for light in lights:
            pose = _quat_to_mat(light["rotation"], light["translation"])
            if light["type"] == "directional":
                scene.add(pyrender.DirectionalLight(color=light["color"], intensity=3.5), pose=pose)
            else:
                scene.add(
                    pyrender.PointLight(color=light["color"], intensity=light["intensity"] * 1.5),
                    pose=pose,
                )

        pose = _quat_to_mat(cam["rotation_quat_xyzw"], cam["position"])
        if cam["projection"] == "orthographic":
            pcam = pyrender.OrthographicCamera(xmag=cam["ortho_mag"], ymag=cam["ortho_mag"],
                                               znear=0.05, zfar=500.0)
        else:
            pcam = pyrender.PerspectiveCamera(yfov=cam["yfov"], znear=0.05)
        scene.add(pcam, pose=pose)

        color, _ = renderer.render(scene)
        png_path = out / f"{cam['name']}.png"
        Image.fromarray(color).save(png_path)
        views.append(
            {
                "name": cam["name"],
                "file": png_path.name,
                "purpose": cam.get("purpose", ""),
                "projection": cam["projection"],
                "resolution": [width, height],
            }
        )
    renderer.delete()

    walkthrough_entry = _render_walkthrough(
        pyrender, trimesh, tm_scene, lights, cameras_doc.get("walkthrough", {}), out,
        min(width, 1280), min(height, 720),
    )

    manifest = {
        "stage": "renders",
        "status": "ok",
        "renderer": "pyrender (GPU rasterization, PBR materials, embedded lights)",
        "resolution": [width, height],
        "views": views,
        "walkthrough": walkthrough_entry
        or {"status": "skipped", "reason": "fewer than two walkthrough waypoints"},
        "pending": {
            "photorealistic_pathtracing": "Blender/Cycles integration pending — "
            "rasterized output is not passed off as photoreal",
        },
    }
    (out / ".." / "renders.json").resolve().write_text(json.dumps(manifest, indent=2))
    return manifest


def _render_walkthrough(pyrender, trimesh, tm_scene, lights, walkthrough: dict,
                        out: Path, width: int, height: int) -> dict | None:
    """Render the door-depth-ordered waypoint path to an H.264 MP4."""
    waypoints = walkthrough.get("waypoints", [])
    if len(waypoints) < 2:
        return None
    import imageio.v2 as iio

    from .cameras import _look_at_quat_xyzw

    # Interior scene (roof hidden), built once; the camera pose animates.
    scene = pyrender.Scene(ambient_light=[0.34, 0.34, 0.36], bg_color=[0.96, 0.96, 0.97])
    for name, geom in tm_scene.geometry.items():
        if name.startswith(("roof_", "roof_parapet")) or not isinstance(geom, trimesh.Trimesh):
            continue
        scene.add(pyrender.Mesh.from_trimesh(geom, smooth=False), name=name)
    for light in lights:
        pose = _quat_to_mat(light["rotation"], light["translation"])
        if light["type"] == "directional":
            scene.add(pyrender.DirectionalLight(color=light["color"], intensity=3.0), pose=pose)
        else:
            scene.add(pyrender.PointLight(color=light["color"],
                                          intensity=light["intensity"] * 1.5), pose=pose)
    cam_node = scene.add(pyrender.PerspectiveCamera(yfov=1.1, znear=0.05), pose=np.eye(4))

    pts = np.array([w["position"] for w in waypoints], dtype=float)
    fps, frames_per_seg = 24, 40
    # Sampled positions with smoothstep easing per segment.
    samples = []
    for i in range(len(pts) - 1):
        for k in range(frames_per_seg):
            t = k / frames_per_seg
            t = t * t * (3 - 2 * t)
            samples.append(pts[i] * (1 - t) + pts[i + 1] * t)
    samples.append(pts[-1])
    samples = np.array(samples)

    renderer = pyrender.OffscreenRenderer(width, height)
    path = out / "walkthrough.mp4"
    writer = iio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    lookahead = 6
    for i, pos in enumerate(samples):
        target = samples[min(i + lookahead, len(samples) - 1)].copy()
        if np.linalg.norm(target - pos) < 0.2:
            target = pos + np.array([0.0, 0.0, 1.0])
        target[1] = 1.35
        q = _look_at_quat_xyzw(pos, target)
        scene.set_pose(cam_node, _quat_to_mat(q, pos))
        color, _ = renderer.render(scene)
        writer.append_data(color)
    writer.close()
    renderer.delete()
    return {
        "file": path.name,
        "fps": fps,
        "frames": len(samples),
        "resolution": [width, height],
        "route": [w["label"] for w in waypoints],
    }


def main() -> int:
    glb, cams, out_dir, w, h = sys.argv[1:6]
    try:
        manifest = render_all(glb, cams, out_dir, int(w), int(h))
        print(f"rendered {len(manifest['views'])} views")
        return 0
    except Exception as exc:  # any GL/driver failure must degrade gracefully
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / ".." / "renders.json").resolve().write_text(
            json.dumps(
                {
                    "stage": "renders",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "note": "rendering requires a GL-capable host; artifacts and the GLB are unaffected",
                },
                indent=2,
            )
        )
        print(f"render failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
