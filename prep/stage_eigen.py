from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trace import Trace

STAGE = "eigen"
TOL = 1e-12


@dataclass
class OBB:
    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray
    eigenvalues: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    checks: dict[str, object] = field(default_factory=dict)

    def corners(self) -> np.ndarray:
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
        )
        return self.center[:, None] + self.axes @ (signs * self.half_extents).T

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "half_extents": self.half_extents.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
        }


@dataclass
class QuadricAxes:
    kind: str
    center: np.ndarray
    axes: np.ndarray
    radii: np.ndarray
    eigenvalues: np.ndarray
    matrix: np.ndarray
    checks: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "radii": self.radii.tolist(),
            "matrix": self.matrix.tolist(),
        }


def sym_eig_descending(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = 0.5 * (m + m.T)
    w, v = np.linalg.eigh(m)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    pivot = np.argmax(np.abs(v), axis=0)
    signs = np.sign(v[pivot, np.arange(v.shape[1])])
    signs[signs == 0] = 1.0
    v = v * signs
    if np.linalg.det(v) < 0.0:
        v[:, -1] = -v[:, -1]
    return w, v


def frame_checks(m: np.ndarray, w: np.ndarray, v: np.ndarray, scale: float) -> dict[str, object]:
    qtq = float(np.abs(v.T @ v - np.eye(3)).max())
    det = float(np.linalg.det(v))
    denom = scale if scale > 0.0 else 1.0
    resid = float(np.abs(m @ v - v * w).max()) / denom
    recon = float(np.abs(v @ np.diag(w) @ v.T - m).max()) / denom
    return {
        "QtQ_error": qtq,
        "QtQ_is_identity": qtq < 1e-10,
        "det": det,
        "det_is_one": abs(det - 1.0) < 1e-10,
        "eig_residual": resid,
        "eig_residual_ok": resid < 1e-10,
        "diag_reconstruction_error": recon,
        "diag_reconstruction_ok": recon < 1e-10,
    }


def point_covariance(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=1)
    x = points - mean[:, None]
    return mean, (x @ x.T) / points.shape[1]


def area_covariance(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = vertices[:, faces[:, 0]].T
    q = vertices[:, faces[:, 1]].T
    r = vertices[:, faces[:, 2]].T
    area = 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)
    total = area.sum()
    if total <= 0.0:
        return point_covariance(vertices[:, np.unique(faces)])
    m = (p + q + r) / 3.0
    mean = (area[:, None] * m).sum(axis=0) / total
    second = (
        np.einsum("i,ij,ik->jk", 9.0 * area / 12.0, m, m)
        + np.einsum("i,ij,ik->jk", area / 12.0, p, p)
        + np.einsum("i,ij,ik->jk", area / 12.0, q, q)
        + np.einsum("i,ij,ik->jk", area / 12.0, r, r)
    )
    cov = second / total - np.outer(mean, mean)
    return mean, 0.5 * (cov + cov.T)


def face_frames(vertices: np.ndarray, faces: np.ndarray, k: int) -> list[np.ndarray]:
    p = vertices[:, faces[:, 0]]
    e = vertices[:, faces[:, 1]] - p
    cross = np.cross(e.T, (vertices[:, faces[:, 2]] - p).T)
    area = np.linalg.norm(cross, axis=1)
    out = []
    for i in np.argsort(area)[::-1][: max(k, 0)]:
        if area[i] <= 0.0:
            break
        n = cross[i] / area[i]
        t = e[:, i] / np.linalg.norm(e[:, i])
        out.append(np.column_stack([t, np.cross(n, t), n]))
    return out


def stage_eigen_obb(
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    trace: Trace | None = None,
    item_name: str = "obb",
    weighting: str = "area",
    min_half_extent: float = 1e-6,
    refine: bool = True,
    refine_faces: int = 4,
) -> OBB:
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape[0] != 3:
        raise ValueError(f"vertices must be 3xN columns, got {v.shape}")
    if faces is not None and len(faces):
        f = np.asarray(faces, dtype=np.int64)
        pts = v[:, np.unique(f)]
        if weighting == "area":
            mean, cov = area_covariance(v, f)
        else:
            mean, cov = point_covariance(pts)
    else:
        pts = v
        mean, cov = point_covariance(pts)

    w, pca_axes = sym_eig_descending(cov)
    frames = [pca_axes]
    labels = ["pca"]
    if refine:
        frames.append(np.eye(3))
        labels.append("world")
        if faces is not None and len(faces):
            frames.extend(face_frames(v, f, refine_faces))
            labels.extend(["face"] * (len(frames) - 2))
    stack = np.stack(frames)
    local = np.einsum("kji,jn->kin", stack, pts - mean[:, None])
    lo = local.min(axis=2)
    hi = local.max(axis=2)
    vols = np.prod(np.maximum(hi - lo, 2.0 * min_half_extent), axis=1)
    best = int(np.argmin(vols))
    if vols[best] >= vols[0] * (1.0 - 1e-9):
        best = 0
    axes = stack[best]
    center = mean + axes @ (0.5 * (lo[best] + hi[best]))
    half = np.maximum(0.5 * (hi[best] - lo[best]), min_half_extent)

    scale = float(np.abs(cov).max())
    checks = frame_checks(cov, w, pca_axes, scale)
    chosen_qtq = float(np.abs(axes.T @ axes - np.eye(3)).max())
    checks["box_frame"] = labels[best]
    checks["box_frame_orthonormal"] = chosen_qtq < 1e-10
    checks["box_frame_det"] = float(np.linalg.det(axes))
    checks["pca_volume"] = float(vols[0])
    inside = np.abs(axes.T @ (pts - center[:, None])) - half[:, None]
    violation = float(max(inside.max(), 0.0))
    extent_scale = float(max(half.max(), 1.0))
    aabb = pts.max(axis=1) - pts.min(axis=1)
    aabb_vol = float(np.prod(np.maximum(aabb, 2.0 * min_half_extent)))
    obb_vol = float(np.prod(2.0 * half))
    checks.update(
        {
            "covariance_symmetric": bool(np.allclose(cov, cov.T, atol=1e-14)),
            "eigenvalues_nonnegative": bool(w.min() >= -1e-12 * max(scale, 1.0)),
            "eigengap_min": float(np.min(np.abs(np.diff(w)))) / max(scale, 1e-300),
            "containment_violation": violation,
            "all_points_inside": violation <= 1e-9 * extent_scale,
            "obb_volume": obb_vol,
            "aabb_volume": aabb_vol,
            "obb_over_aabb": obb_vol / aabb_vol if aabb_vol > 0 else 1.0,
            "n_points": int(pts.shape[1]),
            "weighting": weighting if faces is not None else "points",
        }
    )

    if trace is not None:
        trace.record(STAGE, f"{item_name}/covariance", cov, "3x3 covariance of the mesh cluster")
        trace.record(STAGE, f"{item_name}/eigenvalues", w, "descending eigenvalues of covariance")
        trace.record(
            STAGE,
            f"{item_name}/axes",
            axes,
            "eigenvectors as columns, right handed (PCA frame)",
            checks,
        )
        trace.record(
            STAGE,
            f"{item_name}/box_axes",
            axes,
            f"OBB rotation actually used ({labels[best]}), smallest volume candidate",
        )
        trace.record(
            STAGE,
            f"{item_name}/box",
            np.stack([center, half]),
            "rows: obb center, half extents along axes",
        )

    return OBB(center, axes, half, w, mean, cov, checks)


def cluster_faces(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 64) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return []
    centroids = (vertices[:, f[:, 0]] + vertices[:, f[:, 1]] + vertices[:, f[:, 2]]) / 3.0
    leaves: list[np.ndarray] = []
    stack = [np.arange(len(f))]
    while stack:
        ids = stack.pop()
        if len(ids) <= max_faces:
            leaves.append(ids)
            continue
        c = centroids[:, ids]
        _, cov = point_covariance(c)
        _, v = np.linalg.eigh(cov)
        mid = len(ids) // 2
        order = np.argpartition(v[:, -1] @ c, mid)
        stack.append(ids[order[mid:]])
        stack.append(ids[order[:mid]])
    return leaves


def stage_eigen_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    trace: Trace | None = None,
    item_name: str = "mesh",
    max_cluster_faces: int = 64,
    weighting: str = "area",
) -> tuple[OBB, list[tuple[np.ndarray, OBB]]]:
    whole = stage_eigen_obb(vertices, faces, trace, f"{item_name}/whole", weighting)
    clusters = []
    for i, ids in enumerate(cluster_faces(vertices, faces, max_cluster_faces)):
        obb = stage_eigen_obb(vertices, faces[ids], None, f"{item_name}/cluster{i}", weighting)
        clusters.append((ids, obb))
    if trace is not None and clusters:
        ok = all(o.checks["all_points_inside"] and o.checks["QtQ_is_identity"] for _, o in clusters)
        trace.record(
            STAGE,
            f"{item_name}/cluster_boxes",
            np.stack([np.concatenate([o.center, o.half_extents]) for _, o in clusters]),
            "per cluster obb: cx cy cz hx hy hz",
            {
                "n_clusters": len(clusters),
                "all_clusters_valid": ok,
                "faces_covered": int(sum(len(ids) for ids, _ in clusters)),
            },
        )
    return whole, clusters


def rotation_from_euler_deg(angles: list[float] | np.ndarray) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(angles, dtype=float))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def quadric_from_ellipsoid(
    center: np.ndarray, radii: np.ndarray, rotation: np.ndarray | None = None
) -> np.ndarray:
    c = np.asarray(center, dtype=float)
    r = np.asarray(radii, dtype=float)
    rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    a = rot @ np.diag(1.0 / r**2) @ rot.T
    b = -a @ c
    d = float(c @ a @ c - 1.0)
    q = np.zeros((4, 4))
    q[:3, :3] = a
    q[:3, 3] = b
    q[3, :3] = b
    q[3, 3] = d
    return q


def stage_eigen_quadric(
    q: np.ndarray, trace: Trace | None = None, item_name: str = "quadric"
) -> QuadricAxes:
    q = np.asarray(q, dtype=float)
    if q.shape != (4, 4):
        raise ValueError(f"quadric must be 4x4, got {q.shape}")
    sym_err = float(np.abs(q - q.T).max())
    q = 0.5 * (q + q.T)
    a = q[:3, :3]
    b = q[:3, 3]
    d = q[3, 3]
    w, axes = sym_eig_descending(a)
    scale = float(np.abs(a).max()) or 1.0
    checks = frame_checks(a, w, axes, scale)
    checks["input_symmetry_error"] = sym_err
    checks["input_is_symmetric"] = sym_err < 1e-12 * max(float(np.abs(q).max()), 1.0)

    if np.min(np.abs(w)) <= 1e-12 * scale:
        kind = "degenerate"
        center = np.zeros(3)
        radii = np.full(3, np.inf)
    else:
        center = -np.linalg.solve(a, b)
        s = float(center @ a @ center - d)
        if w.min() > 0 and s > 0:
            kind = "ellipsoid"
        elif w.max() < 0 and s < 0:
            kind = "ellipsoid"
            w, s = -w, -s
        elif (w > 0).all() or (w < 0).all():
            kind = "point" if abs(s) <= 1e-12 * scale else "empty"
        else:
            kind = "hyperboloid"
        with np.errstate(invalid="ignore", divide="ignore"):
            radii = np.sqrt(np.abs(s / w))
        if kind == "ellipsoid":
            order = np.argsort(radii)[::-1]
            w = w[order]
            axes = axes[:, order]
            radii = radii[order]
            if np.linalg.det(axes) < 0:
                axes[:, -1] = -axes[:, -1]
            if np.allclose(radii, radii[0], rtol=1e-12):
                kind = "sphere"

    if kind in ("ellipsoid", "sphere"):
        rebuilt = quadric_from_ellipsoid(center, radii, axes)
        norm = q / abs(q[3, 3]) if abs(q[3, 3]) > 0 else q
        ref = rebuilt / abs(rebuilt[3, 3]) if abs(rebuilt[3, 3]) > 0 else rebuilt
        if np.sign(norm[0, 0]) != np.sign(ref[0, 0]):
            ref = -ref
        err = float(np.abs(norm - ref).max())
        checks["rebuild_error"] = err
        checks["rebuild_ok"] = err < 1e-9
        surf = center[:, None] + axes @ np.diag(radii)
        vals = np.einsum(
            "ji,jk,ki->i", np.vstack([surf, np.ones(3)]), q, np.vstack([surf, np.ones(3)])
        )
        checks["axis_tips_on_surface"] = bool(np.all(np.abs(vals) < 1e-9 * max(abs(d), 1.0) + 1e-9))
    checks["kind"] = kind

    if trace is not None:
        trace.record(STAGE, f"{item_name}/Q", q, "homogeneous symmetric quadric matrix")
        trace.record(STAGE, f"{item_name}/eigenvalues", w, "eigenvalues of upper 3x3 block")
        trace.record(
            STAGE,
            f"{item_name}/axes",
            axes,
            "principal axes as columns from symmetric diagonalization",
            checks,
        )
        trace.record(
            STAGE,
            f"{item_name}/center_radii",
            np.stack([center, radii]),
            "rows: center, semi axis lengths",
        )

    return QuadricAxes(kind, center, axes, radii, w, q, checks)


def intersect_ellipsoid(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    radii: np.ndarray,
    t_min: float = 1e-6,
) -> np.ndarray:
    o = np.atleast_2d(origin).astype(float)
    dvec = np.atleast_2d(direction).astype(float)
    to_unit = np.diag(1.0 / np.asarray(radii)) @ np.asarray(axes).T
    ou = (o - center) @ to_unit.T
    du = dvec @ to_unit.T
    a = np.einsum("ij,ij->i", du, du)
    b = np.einsum("ij,ij->i", ou, du)
    c = np.einsum("ij,ij->i", ou, ou) - 1.0
    disc = b * b - a * c
    t = np.full(len(o), np.inf)
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / a
    t1 = (-b + sq) / a
    t = np.where(hit & (t0 > t_min), t0, t)
    t = np.where(hit & ~(t0 > t_min) & (t1 > t_min), t1, t)
    return t
