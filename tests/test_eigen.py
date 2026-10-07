import time

import numpy as np
import pytest

from prep.stage_eigen import (
    cluster_faces,
    intersect_ellipsoid,
    quadric_from_ellipsoid,
    rotation_from_euler_deg,
    stage_eigen_mesh,
    stage_eigen_obb,
    stage_eigen_quadric,
)
from prep.trace import Trace


def random_rotation(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def box_mesh(half):
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    v = (s * half).T
    f = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    return v, f


def test_obb_recovers_rotated_box():
    rng = np.random.default_rng(1)
    for _ in range(20):
        half = np.sort(rng.uniform(0.2, 3.0, 3))[::-1]
        r = random_rotation(rng)
        c = rng.standard_normal(3)
        v, f = box_mesh(half)
        v = r @ v + c[:, None]
        obb = stage_eigen_obb(v, f)
        assert obb.checks["QtQ_is_identity"]
        assert obb.checks["det_is_one"]
        assert obb.checks["all_points_inside"]
        assert np.allclose(obb.center, c, atol=1e-9)
        assert np.allclose(np.sort(obb.half_extents), np.sort(half), atol=1e-6)


def test_obb_points_match_numpy_reference():
    rng = np.random.default_rng(2)
    r = random_rotation(rng)
    pts = r @ (rng.standard_normal((3, 4000)) * np.array([[5.0], [2.0], [0.5]]))
    obb = stage_eigen_obb(pts, refine=False)
    cov = np.cov(pts, bias=True)
    w, v = np.linalg.eigh(cov)
    assert np.allclose(obb.eigenvalues, w[::-1])
    for k in range(3):
        assert abs(abs(obb.axes[:, k] @ v[:, 2 - k]) - 1.0) < 1e-10
    assert obb.checks["eig_residual_ok"]
    assert obb.checks["diag_reconstruction_ok"]


def test_area_weighting_ignores_tessellation_density():
    half = np.array([2.0, 1.0, 0.5])
    v, f = box_mesh(half)
    keep = np.array([i for i in range(12) if i not in (2, 3)])
    n = 20
    ys, zs = np.meshgrid(np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1), indexing="ij")
    grid = np.stack([np.full(ys.size, half[0]), ys.ravel() * half[1], zs.ravel() * half[2]])
    idx = np.arange(ys.size).reshape(n + 1, n + 1) + 8
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    gf = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    dv = np.hstack([v, grid])
    df = np.vstack([f[keep], gf])
    obb_area = stage_eigen_obb(dv, df, weighting="area", refine=False)
    obb_pts = stage_eigen_obb(dv, df, weighting="points", refine=False)
    ref_area = stage_eigen_obb(v, f, weighting="area", refine=False)
    assert np.allclose(obb_area.mean, 0.0, atol=1e-12)
    assert np.allclose(obb_area.covariance, ref_area.covariance, atol=1e-12)
    assert obb_pts.mean[0] > 1.0
    assert np.allclose(obb_area.half_extents, half, atol=1e-9)


def test_cube_degenerate_eigenspace_still_tight():
    v, f = box_mesh(np.array([1.0, 1.0, 1.0]))
    r = rotation_from_euler_deg([10, 25, 40])
    obb = stage_eigen_obb(r @ v, f)
    assert obb.checks["eigengap_min"] < 1e-9
    assert np.allclose(obb.half_extents, 1.0, atol=1e-9)
    assert obb.checks["box_frame"] != "pca" or np.isclose(obb.checks["obb_volume"], 8.0)


def test_flat_mesh_gets_padded_box():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float).T
    obb = stage_eigen_obb(v, np.array([[0, 1, 2], [0, 2, 3]]))
    assert obb.half_extents.min() > 0.0
    assert obb.checks["all_points_inside"]


def test_clusters_partition_faces():
    rng = np.random.default_rng(7)
    v = rng.standard_normal((3, 3000))
    f = rng.integers(0, 3000, size=(5000, 3))
    leaves = cluster_faces(v, f, 64)
    allids = np.sort(np.concatenate(leaves))
    assert np.array_equal(allids, np.arange(5000))
    assert max(len(x) for x in leaves) <= 64


def test_mesh_trace_records():
    v, f = box_mesh(np.array([1.0, 2.0, 3.0]))
    tr = Trace()
    _whole, clusters = stage_eigen_mesh(v, f, tr, "box", max_cluster_faces=4)
    names = {r["name"] for r in tr.records}
    assert "box/whole/axes" in names
    assert "box/cluster_boxes" in names
    assert all(r["stage"] == "eigen" for r in tr.records)
    assert sum(len(ids) for ids, _ in clusters) == 12


def test_quadric_roundtrip_and_classify():
    rng = np.random.default_rng(11)
    for _ in range(50):
        r = random_rotation(rng)
        radii = rng.uniform(0.1, 4.0, 3)
        c = rng.standard_normal(3) * 3
        q = quadric_from_ellipsoid(c, radii, r)
        q = q * rng.uniform(-5, 5)
        res = stage_eigen_quadric(q)
        assert res.kind == "ellipsoid"
        assert np.allclose(res.center, c, atol=1e-8)
        assert np.allclose(res.radii, np.sort(radii)[::-1], rtol=1e-8)
        assert res.checks["QtQ_is_identity"]
        assert res.checks["det_is_one"]
        assert res.checks["rebuild_ok"]
        assert res.checks["axis_tips_on_surface"]


def test_quadric_kinds():
    assert stage_eigen_quadric(quadric_from_ellipsoid([1, 2, 3], [2, 2, 2])).kind == "sphere"
    hyper = np.diag([1.0, 1.0, -1.0, -1.0])
    assert stage_eigen_quadric(hyper).kind == "hyperboloid"
    empty = np.diag([1.0, 1.0, 1.0, 1.0])
    assert stage_eigen_quadric(empty).kind == "empty"
    cyl = np.diag([1.0, 1.0, 0.0, -1.0])
    assert stage_eigen_quadric(cyl).kind == "degenerate"


def test_quadric_rejects_bad_shape():
    with pytest.raises(ValueError):
        stage_eigen_quadric(np.eye(3))


def test_intersection_matches_quadric_roots():
    rng = np.random.default_rng(13)
    r = random_rotation(rng)
    radii = np.array([1.5, 0.7, 0.4])
    c = np.array([0.3, -0.2, -2.0])
    q = quadric_from_ellipsoid(c, radii, r)
    res = stage_eigen_quadric(q)
    o = np.tile([0.0, 0.0, 3.0], (2000, 1))
    target = c + rng.standard_normal((2000, 3)) * 0.8
    d = target - o
    d /= np.linalg.norm(d, axis=1)[:, None]
    t = intersect_ellipsoid(o, d, res.center, res.axes, res.radii)
    oh = np.hstack([o, np.ones((2000, 1))])
    dh = np.hstack([d, np.zeros((2000, 1))])
    a = np.einsum("ni,ij,nj->n", dh, q, dh)
    b = np.einsum("ni,ij,nj->n", oh, q, dh)
    cc = np.einsum("ni,ij,nj->n", oh, q, oh)
    disc = b * b - a * cc
    ref = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
    assert np.isfinite(t).sum() > 100
    assert np.allclose(t, ref, rtol=1e-9)
    hit = np.isfinite(t)
    p = o[hit] + t[hit, None] * d[hit]
    ph = np.hstack([p, np.ones((hit.sum(), 1))])
    assert np.abs(np.einsum("ni,ij,nj->n", ph, q, ph)).max() < 1e-9


def test_large_mesh_eigen_is_fast():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((3, 100_000))
    f = rng.integers(0, 100_000, size=(200_000, 3))
    t = time.perf_counter()
    stage_eigen_mesh(v, f, None, "big", max_cluster_faces=256)
    assert time.perf_counter() - t < 10.0
