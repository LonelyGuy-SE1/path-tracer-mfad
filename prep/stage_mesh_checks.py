from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trace import Trace

STAGE = "mesh_checks"

REASONS = (
    "bad_index",
    "nonfinite_vertex",
    "repeated_index",
    "coincident_vertices",
    "rank0_collapsed",
    "rank1_collinear",
    "duplicate_face",
)


@dataclass
class MeshCheckResult:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    face_rank: np.ndarray
    keep: np.ndarray
    reason: np.ndarray
    vertex_map: np.ndarray
    affine_rank: int
    affine_singular_values: np.ndarray
    basis_vertices: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)
    checks: dict[str, object] = field(default_factory=dict)


def as_points(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] not in (3, 4):
        raise ValueError(f"vertices must be 3xN or 4xN columns, got {v.shape}")
    if v.shape[0] == 4:
        w = v[3]
        safe = np.where(np.abs(w) > 0.0, w, 1.0)
        v = v[:3] / safe
    return np.ascontiguousarray(v.T)


def edge_singular_values(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    e1 = p1 - p0
    e2 = p2 - p0
    a = np.einsum("ij,ij->i", e1, e1)
    c = np.einsum("ij,ij->i", e2, e2)
    b = np.einsum("ij,ij->i", e1, e2)
    cross = np.cross(e1, e2)
    det_g = np.einsum("ij,ij->i", cross, cross)
    half = 0.5 * (a + c)
    lam_max = half + np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_min = np.where(lam_max > 0.0, det_g / lam_max, 0.0)
    return np.sqrt(lam_max), np.sqrt(np.maximum(lam_min, 0.0)), cross


def face_rank(
    sigma_max: np.ndarray, sigma_min: np.ndarray, abs_tol: float, rel_tol: float
) -> np.ndarray:
    rank = np.full(sigma_max.shape, 2, dtype=np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sigma_max > 0.0, sigma_min / sigma_max, 0.0)
    rank[ratio <= rel_tol] = 1
    rank[sigma_max <= abs_tol] = 0
    return rank


def weld(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    if tol <= 0.0 or len(points) == 0:
        return points, np.arange(len(points))
    keys = np.floor(points / tol + 0.5).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))
    return points[first[order]], rank_of[inverse]


def affine_basis(points: np.ndarray, rank: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    centroid = points.mean(axis=0)
    chosen = [int(np.argmax(np.einsum("ij,ij->i", points - centroid, points - centroid)))]
    if rank == 0:
        return np.array(chosen, dtype=np.int64)
    origin = points[chosen[0]]
    rel = points - origin
    directions: list[np.ndarray] = []
    for _ in range(rank):
        resid = rel.copy()
        for d in directions:
            resid -= np.outer(resid @ d, d)
        dist = np.einsum("ij,ij->i", resid, resid)
        idx = int(np.argmax(dist))
        chosen.append(idx)
        directions.append(resid[idx] / np.sqrt(dist[idx]))
    return np.array(chosen, dtype=np.int64)


def affine_rank(points: np.ndarray, rel_tol: float) -> tuple[int, np.ndarray]:
    if len(points) < 2:
        return 0, np.zeros(3)
    centered = points - points.mean(axis=0)
    sv = np.linalg.svd(centered.T @ centered, compute_uv=False)
    sv = np.sqrt(np.maximum(sv, 0.0))
    if sv[0] <= 0.0:
        return 0, sv
    return int(np.count_nonzero(sv > rel_tol * sv[0])), sv


def stage_mesh_checks(
    vertices: np.ndarray,
    faces: np.ndarray,
    trace: Trace | None = None,
    item_name: str = "mesh",
    weld_tol: float = 1e-9,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-8,
    drop_duplicate_faces: bool = True,
) -> MeshCheckResult:
    points = as_points(vertices)
    f = np.asarray(faces)
    if f.size == 0:
        f = np.zeros((0, 3), dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be Mx3 triangle indices, got {f.shape}")
    f = f.astype(np.int64)
    n_in_v = len(points)
    n_in_f = len(f)

    reason = np.full(n_in_f, -1, dtype=np.int8)

    def mark(mask: np.ndarray, code: str) -> None:
        reason[(reason < 0) & mask] = REASONS.index(code)

    mark(((f < 0) | (f >= n_in_v)).any(axis=1), "bad_index")
    fc = np.clip(f, 0, max(n_in_v - 1, 0))

    finite_v = np.isfinite(points).all(axis=1) if n_in_v else np.zeros(0, bool)
    if n_in_v:
        mark(~finite_v[fc].all(axis=1), "nonfinite_vertex")
    mark(
        (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]),
        "repeated_index",
    )

    clean_pts = np.where(finite_v[:, None], points, 0.0) if n_in_v else points
    welded, vmap = weld(clean_pts, weld_tol)
    wf = vmap[fc] if n_in_v else fc
    mark(
        (wf[:, 0] == wf[:, 1]) | (wf[:, 1] == wf[:, 2]) | (wf[:, 0] == wf[:, 2]),
        "coincident_vertices",
    )

    if len(welded):
        s_max, s_min, cross = edge_singular_values(
            welded[wf[:, 0]], welded[wf[:, 1]], welded[wf[:, 2]]
        )
    else:
        s_max = s_min = np.zeros(n_in_f)
        cross = np.zeros((n_in_f, 3))
    rank = face_rank(s_max, s_min, abs_tol, rel_tol)
    mark(rank == 0, "rank0_collapsed")
    mark(rank == 1, "rank1_collinear")

    if drop_duplicate_faces and n_in_f:
        alive = np.flatnonzero(reason < 0)
        if len(alive):
            keyed = np.sort(wf[alive], axis=1)
            _, first = np.unique(keyed, axis=0, return_index=True)
            dup = np.ones(len(alive), dtype=bool)
            dup[first] = False
            dmask = np.zeros(n_in_f, dtype=bool)
            dmask[alive[dup]] = True
            mark(dmask, "duplicate_face")

    keep = reason < 0
    kept = wf[keep]
    used = np.unique(kept) if len(kept) else np.zeros(0, dtype=np.int64)
    compact = np.full(len(welded), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    out_pts = welded[used]
    out_faces = compact[kept]
    vertex_map = compact[vmap] if n_in_v else np.zeros(0, dtype=np.int64)
    if n_in_v:
        vertex_map[~finite_v] = -1

    kept_cross = cross[keep]
    norms = np.linalg.norm(kept_cross, axis=1)
    normals = kept_cross / np.where(norms > 0.0, norms, 1.0)[:, None]
    areas = 0.5 * norms

    a_rank, a_sv = affine_rank(out_pts, rel_tol)
    basis = affine_basis(out_pts, a_rank)

    counts = {code: int(np.count_nonzero(reason == i)) for i, code in enumerate(REASONS)}
    unit_err = float(np.abs(np.linalg.norm(normals, axis=1) - 1.0).max()) if len(normals) else 0.0
    kept_rank_ok = bool(np.all(rank[keep] == 2))
    checks: dict[str, object] = {
        "faces_in": n_in_f,
        "faces_out": len(out_faces),
        "faces_dropped": int(n_in_f - len(out_faces)),
        "vertices_in": n_in_v,
        "vertices_out": len(out_pts),
        "vertices_merged": int(n_in_v - len(welded)),
        "vertices_unreferenced": int(len(welded) - len(used)),
        **{f"dropped_{k}": v for k, v in counts.items()},
        "all_kept_faces_rank2": kept_rank_ok,
        "normals_unit_error": unit_err,
        "normals_are_unit": unit_err < 1e-12,
        "indices_in_range": bool(len(out_faces) == 0 or out_faces.max() < len(out_pts)),
        "affine_rank": a_rank,
        "mesh_is_flat": a_rank < 3,
        "rel_tol": rel_tol,
        "weld_tol": weld_tol,
    }

    if trace is not None:
        trace.record(
            STAGE,
            f"{item_name}/face_singular_values",
            np.stack([s_max, s_min]),
            "per face singular values of edge matrix [v1-v0, v2-v0]; rows sigma_max, sigma_min",
        )
        trace.record(
            STAGE,
            f"{item_name}/face_rank",
            rank.astype(np.int64),
            "numerical rank of each triangle edge matrix (2 = valid)",
        )
        trace.record(
            STAGE,
            f"{item_name}/drop_reason",
            reason.astype(np.int64),
            "-1 kept, else index into " + ",".join(REASONS),
        )
        trace.record(
            STAGE,
            f"{item_name}/affine_singular_values",
            a_sv,
            "singular values of centred vertex scatter; count above tol is the affine rank",
        )
        trace.record(
            STAGE,
            f"{item_name}/basis_vertices",
            out_pts[basis].T if len(basis) else np.zeros((3, 0)),
            "affinely independent vertices selected greedily (columns)",
            checks,
        )

    return MeshCheckResult(
        vertices=out_pts.T.copy(),
        faces=out_faces,
        normals=normals.T.copy(),
        areas=areas,
        face_rank=rank,
        keep=keep,
        reason=reason,
        vertex_map=vertex_map,
        affine_rank=a_rank,
        affine_singular_values=a_sv,
        basis_vertices=basis,
        counts=counts,
        checks=checks,
    )
