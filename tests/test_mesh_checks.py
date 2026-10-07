import time

import numpy as np
import pytest

from prep.stage_mesh_checks import REASONS, stage_mesh_checks
from prep.trace import Trace, load_trace


def cube():
    v = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    ).T
    f = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return v, f


def reason_of(res, i):
    return REASONS[res.reason[i]] if res.reason[i] >= 0 else "kept"


def test_clean_cube_untouched():
    v, f = cube()
    res = stage_mesh_checks(v, f)
    assert res.keep.all()
    assert res.faces.shape == (12, 3)
    assert res.vertices.shape == (3, 8)
    assert res.affine_rank == 3
    assert np.allclose(res.areas.sum(), 24.0)
    assert res.checks["normals_are_unit"]


def test_each_degeneracy_detected():
    v, f = cube()
    extra_v = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [1, 1, 1 + 1e-12], [np.nan, 0, 0]]).T
    v = np.hstack([v, extra_v])
    bad = np.array(
        [
            [0, 0, 1],
            [8, 9, 10],
            [6, 11, 5],
            [0, 1, 99],
            [0, 1, 12],
            [2, 0, 3],
            [3, 3, 3],
        ]
    )
    res = stage_mesh_checks(v, np.vstack([f, bad]))
    got = [reason_of(res, 12 + i) for i in range(len(bad))]
    assert got == [
        "repeated_index",
        "rank1_collinear",
        "coincident_vertices",
        "bad_index",
        "nonfinite_vertex",
        "duplicate_face",
        "repeated_index",
    ]


def test_coincident_after_weld():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1e-13], [0, 1, 0]], dtype=float).T
    f = np.array([[0, 1, 3], [0, 1, 2]])
    res = stage_mesh_checks(v, f, weld_tol=1e-9)
    assert reason_of(res, 1) == "coincident_vertices"
    assert res.vertices.shape[1] == 3
    assert res.checks["vertices_merged"] == 1


def test_sliver_is_rank1_relative():
    v = np.array([[0, 0, 0], [1000, 0, 0], [500, 1e-7, 0]], dtype=float).T
    res = stage_mesh_checks(v, np.array([[0, 1, 2]]))
    assert reason_of(res, 0) == "rank1_collinear"
    tiny = np.array([[0, 0, 0], [1e-6, 0, 0], [0, 1e-6, 0]]).T
    res = stage_mesh_checks(tiny, np.array([[0, 1, 2]]), weld_tol=0.0)
    assert reason_of(res, 0) == "kept"


def test_rank_matches_numpy_svd():
    rng = np.random.default_rng(3)
    p = rng.standard_normal((500, 3, 3))
    p[::5, 2] = p[::5, 0] + 0.3 * (p[::5, 1] - p[::5, 0])
    v = p.reshape(-1, 3).T
    f = np.arange(1500).reshape(-1, 3)
    res = stage_mesh_checks(v, f, weld_tol=0.0)
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)
    sv = np.linalg.svd(e, compute_uv=False)
    ref = np.where(sv[:, 1] / sv[:, 0] > 1e-8, 2, 1)
    assert np.array_equal(res.face_rank, ref)


def test_flat_mesh_rank_and_basis():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float).T
    res = stage_mesh_checks(v, np.array([[0, 1, 2], [0, 2, 3]]))
    assert res.affine_rank == 2
    assert res.checks["mesh_is_flat"]
    b = res.vertices[:, res.basis_vertices]
    assert np.linalg.matrix_rank(b[:, 1:] - b[:, :1]) == 2


def test_homogeneous_input_and_trace(tmp_path):
    v, f = cube()
    h = np.vstack([2.0 * v, 2.0 * np.ones(8)])
    tr = Trace(inline_limit=4)
    res = stage_mesh_checks(h, f, tr, "cube")
    assert np.allclose(res.vertices.min(axis=1), -1.0)
    out = tr.save_json(tmp_path / "trace_mesh.json")
    recs = load_trace(out)
    assert {r["stage"] for r in recs} == {"mesh_checks"}
    rank = next(r for r in recs if r["name"] == "cube/face_rank")
    assert np.array_equal(rank["array"], res.face_rank)
    last = recs[-1]["checks"]
    assert last["all_kept_faces_rank2"] is True


def test_vertex_map_roundtrip():
    v, f = cube()
    v = np.hstack([v, v[:, :2]])
    f2 = f.copy()
    f2[0] = [8, 3, 2]
    res = stage_mesh_checks(v, f2)
    mapped = res.vertex_map[f2[res.keep]]
    assert np.array_equal(np.sort(mapped, axis=1), np.sort(res.faces, axis=1))


@pytest.mark.parametrize("n", [200_000])
def test_large_mesh_is_fast(n):
    rng = np.random.default_rng(0)
    v = rng.standard_normal((3, n))
    f = rng.integers(0, n, size=(2 * n, 3))
    t = time.perf_counter()
    res = stage_mesh_checks(v, f)
    dt = time.perf_counter() - t
    assert res.faces.shape[0] > 0
    assert dt < 3.0
