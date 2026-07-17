"""Generate a synthetic floor plan for tests and demos.

Layout: an outer shell with one vertical partition and one horizontal
partition on the right half — three rooms, each connected by a door gap —
plus thin annotation linework and text that the wall detector must ignore.

Run:  python -m app.devtools.sample_plan [out.png]
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

WALL = 14  # px wall thickness


def draw_sample_plan(width: int = 1400, height: int = 1000) -> np.ndarray:
    img = np.full((height, width), 255, np.uint8)

    x0, y0, x1, y1 = 60, 60, width - 60, height - 60
    black = 0

    def hwall(xa: int, xb: int, y: int) -> None:
        cv2.rectangle(img, (xa, y), (xb, y + WALL), black, -1)

    def vwall(x: int, ya: int, yb: int) -> None:
        cv2.rectangle(img, (x, ya), (x + WALL, yb), black, -1)

    # Outer shell.
    hwall(x0, x1, y0)
    hwall(x0, x1, y1 - WALL)
    vwall(x0, y0, y1)
    vwall(x1 - WALL, y0, y1)

    # Vertical partition at x=760 with a 90px door gap in the middle.
    door = 90
    mid_y = (y0 + y1) // 2
    vwall(760, y0, mid_y - door // 2)
    vwall(760, mid_y + door // 2, y1)

    # Horizontal partition across the right half with a door gap.
    gap_x = 1050
    hwall(760, gap_x - door // 2, 520)
    hwall(gap_x + door // 2, x1, 520)

    # Annotation noise: dimension lines, furniture symbol, labels (thin strokes).
    cv2.line(img, (x0, 30), (x1, 30), black, 1)
    cv2.rectangle(img, (150, 700), (330, 830), black, 2)
    cv2.circle(img, (1150, 750), 45, black, 2)
    cv2.putText(img, "LIVING ROOM", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, black, 2)
    cv2.putText(img, "BEDROOM", (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, black, 2)
    cv2.putText(img, "BATH", (1000, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.9, black, 2)

    return img


def sample_plan_png() -> bytes:
    ok, buf = cv2.imencode(".png", draw_sample_plan())
    assert ok
    return buf.tobytes()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_plan.png"
    cv2.imwrite(out, draw_sample_plan())
    print(f"wrote {out}")
