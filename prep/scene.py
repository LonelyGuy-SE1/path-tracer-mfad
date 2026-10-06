from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .obj import load_obj
from .stage_eigen import (
    quadric_from_ellipsoid,
    rotation_from_euler_deg,
    stage_eigen_mesh,
    stage_eigen_quadric,
)
from .stage_mesh_checks import stage_mesh_checks
from .trace import BinStore, Trace, to_jsonable

HANDLED = {"meshes", "quadrics"}


def mesh_arrays(spec: dict[str, Any], base: Path) -> tuple[np.ndarray, np.ndarray]:
    if "file" in spec:
        v, f = load_obj(base / spec["file"])
    else:
        v = np.asarray(spec["vertices"], dtype=float).T
        f = np.asarray(spec["faces"], dtype=np.int64)
    if "transform" in spec:
        m = np.asarray(spec["transform"], dtype=float)
        h = m @ np.vstack([v, np.ones(v.shape[1])])
        v = h[:3] / h[3]
    return v, f


def quadric_matrix(spec: dict[str, Any]) -> np.ndarray:
    if "matrix" in spec:
        return np.asarray(spec["matrix"], dtype=float)
    center = spec.get("center", [0.0, 0.0, 0.0])
    if spec.get("type", "ellipsoid") == "sphere":
        radii = [float(spec["radius"])] * 3
    else:
        radii = spec["radii"]
    if "rotation" in spec:
        rot = np.asarray(spec["rotation"], dtype=float)
    else:
        rot = rotation_from_euler_deg(spec.get("rotation_deg", [0.0, 0.0, 0.0]))
    return quadric_from_ellipsoid(center, radii, rot)


def prepare_scene(
    scene: dict[str, Any],
    base: Path,
    trace: Trace | None = None,
    max_cluster_faces: int = 64,
    store: BinStore | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    store = store or BinStore()
    timings = {"mesh_checks": 0.0, "eigen": 0.0}
    out: dict[str, Any] = {k: v for k, v in scene.items() if k not in HANDLED}
    out["meshes"] = []
    out["quadrics"] = []

    for i, spec in enumerate(scene.get("meshes", [])):
        name = spec.get("name", f"mesh{i}")
        v, f = mesh_arrays(spec, base)
        t0 = time.perf_counter()
        mc = stage_mesh_checks(
            v,
            f,
            trace,
            name,
            weld_tol=spec.get("weld_tol", 1e-9),
            rel_tol=spec.get("rank_tol", 1e-8),
        )
        t1 = time.perf_counter()
        if len(mc.faces) == 0:
            timings["mesh_checks"] += t1 - t0
            out["meshes"].append({"name": name, "skipped": True, "checks": to_jsonable(mc.checks)})
            continue
        whole, clusters = stage_eigen_mesh(mc.vertices, mc.faces, trace, name, max_cluster_faces)
        t2 = time.perf_counter()
        timings["mesh_checks"] += t1 - t0
        timings["eigen"] += t2 - t1

        order = np.concatenate([ids for ids, _ in clusters])
        ranges = np.cumsum([0] + [len(ids) for ids, _ in clusters])
        entry = {k: v for k, v in spec.items() if k not in ("vertices", "faces", "file")}
        entry.update(
            {
                "name": name,
                "vertices": store.pack(mc.vertices),
                "vertices_shape": list(mc.vertices.shape),
                "faces": store.pack(mc.faces[order]),
                "faces_shape": [len(order), 3],
                "normals": store.pack(mc.normals[:, order]),
                "obb": whole.to_dict(),
                "clusters": [
                    {"begin": int(ranges[k]), "end": int(ranges[k + 1]), **clusters[k][1].to_dict()}
                    for k in range(len(clusters))
                ],
                "checks": to_jsonable(mc.checks),
            }
        )
        out["meshes"].append(entry)

    for i, spec in enumerate(scene.get("quadrics", [])):
        name = spec.get("name", f"quadric{i}")
        t0 = time.perf_counter()
        qa = stage_eigen_quadric(quadric_matrix(spec), trace, name)
        timings["eigen"] += time.perf_counter() - t0
        entry = {k: v for k, v in spec.items() if k not in ("matrix",)}
        entry.update({"name": name, **qa.to_dict(), "checks": to_jsonable(qa.checks)})
        out["quadrics"].append(entry)

    return out, timings


def prepare_file(
    scene_path: str | Path,
    out_path: str | Path,
    trace_path: str | Path | None = None,
    max_cluster_faces: int = 64,
) -> dict[str, float]:
    scene_path = Path(scene_path)
    out_path = Path(out_path)
    scene = json.loads(scene_path.read_text())
    trace = Trace() if trace_path else None
    store = BinStore()
    prepared, timings = prepare_scene(scene, scene_path.parent, trace, max_cluster_faces, store)
    bin_path = out_path.with_suffix(".bin")
    if store.write(bin_path):
        prepared["binary"] = bin_path.name
    out_path.write_text(json.dumps(to_jsonable(prepared), indent=2))
    if trace is not None and trace_path is not None:
        trace.save_json(trace_path)
    return timings
