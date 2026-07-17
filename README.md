# Architect SaaS — 2D Floor Plan → 3D BIM Reconstruction

Converts architectural floor plan images into BIM-ready 3D models. This
repository contains a **working end-to-end vertical slice** of the platform
blueprint: upload → preprocessing → structure detection → vectorization →
room graph → classification → 3D reconstruction → validation → GLB export,
with a FastAPI backend and a Next.js + React Three Fiber viewer.

![pipeline](samples/sample_plan.png)

## Quick start

```bash
# Backend (Python 3.12+)
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests        # 12 tests, all green
.venv/bin/python -m uvicorn app.main:app --port 8000

# Frontend
cd frontend
npm install && npm run dev              # http://localhost:3000

# Or everything at once
docker compose up --build
```

Try it: drop `samples/sample_plan.png` onto the page, or generate a fresh one
with `python -m app.devtools.sample_plan`.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/plans` | Upload a PNG/JPEG plan (multipart `file`, optional `?meters_per_px=`). Returns `202 {job_id}`. |
| GET | `/api/v1/jobs/{id}` | Job status, detected rooms, adjacency graph, validation report. |
| GET | `/api/v1/jobs/{id}/model.glb` | The reconstructed 3D model. |
| GET | `/health` | Liveness probe (public, unauthenticated). |

## Architecture

```
backend/app/pipeline/        one module per stage, typed contracts between them
  preprocess.py   decode + binarize (pixel caps against decompression bombs)
  detect.py       wall extraction  ← ML plug-in point (YOLO/RT-DETR/SAM2)
  vectorize.py    masks → Shapely polygons; room segmentation
  graph.py        room-connectivity graph (doors = openings)
  classify.py     room labels      ← ML plug-in point (GraphSAGE/GCN)
  ocr.py          text/scale       ← ML plug-in point (PaddleOCR/TrOCR)
  reconstruct.py  2D → 3D (Trimesh), pixel → meter scaling
  validate.py     mesh integrity, scale plausibility, room reachability
  runner.py       orchestrator — pure function, Celery-ready
```

Each stage consumes and returns the dataclasses in `pipeline/types.py`, so the
classical-CV MVP detectors can be swapped for trained models without touching
the orchestration, API, or frontend.

## Security posture

- **Uploads**: magic-byte sniffing (not client MIME), hard size cap (20 MB),
  pixel-count cap (40 MP) against decompression bombs.
- **Filesystem**: artifact paths are built only from server-generated UUIDs;
  no user-controlled value ever reaches a path.
- **API**: optional key auth (`ARCH_API_KEY`, constant-time compare), per-IP
  token-bucket rate limiting, hardening headers, CORS locked to the frontend
  origin, GET/POST only.
- **Errors**: internal failures log the traceback server-side and return a
  generic message — no stack traces or paths leak to clients.
- **Containers**: both images run as unprivileged users; backend data on a
  dedicated volume.

Production additions to make before exposure to the internet: TLS termination,
real identity (OIDC) + per-tenant quotas instead of a shared API key, object
storage (MinIO/R2) for artifacts, and Postgres instead of the SQLite job store
(the store is isolated behind `app/store.py` for exactly this swap).

## Roadmap (from the platform blueprint)

| Module | Status | Where it plugs in |
|---|---|---|
| Wall/room extraction (classical CV) | ✅ working | `detect.py`, `vectorize.py` |
| Room connectivity graph | ✅ working | `graph.py` |
| 3D reconstruction + GLB export | ✅ working | `reconstruct.py` |
| Validation suite | ✅ working | `validate.py` |
| Web viewer (R3F) | ✅ working | `frontend/` |
| Doors/windows/stairs detectors (YOLO/SAM2) | ⬜ | `detect.py` contract |
| OCR: room names, dimensions, scale | ⬜ | `ocr.py` contract |
| GNN room classifier | ⬜ | `classify.py` contract |
| IFC / USD / FBX export | ⬜ | `reconstruct.py` (IfcOpenShell) |
| Celery + RabbitMQ workers | ⬜ | `runner.py` is already a pure function |
| Furniture placement, materials, costing | ⬜ | new stages after `reconstruct` |
