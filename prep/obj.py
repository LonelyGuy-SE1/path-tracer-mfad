from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    k = int(tok.split("/")[0])
                    idx.append(k - 1 if k > 0 else len(verts) + k)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3).T
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return v, f


def save_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as fh:
        for x, y, z in vertices.T:
            fh.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in faces + 1:
            fh.write(f"f {a} {b} {c}\n")
