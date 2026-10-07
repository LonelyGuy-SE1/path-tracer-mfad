import os
from pathlib import Path

import numpy as np
import pytest

from prep.trace import load_trace

TRACE_PATH = Path("trace_projection.json")


def projector_numpy(a: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the column space of A, computed from the SVD pseudo-inverse.

    Deliberately a different route than the C++ A (A^T A)^-1 A^T so the two can disagree.
    """
    a = np.asarray(a, dtype=float)
    return a @ np.linalg.pinv(a)


def project_onto_normal_numpy(v: np.ndarray, normal: np.ndarray) -> dict:
    """NumPy reference for stage_projection_normal, using the Householder matrix for reflection."""
    n = normal / np.linalg.norm(normal)
    p_n = np.outer(n, n)
    p_t = np.eye(3) - p_n
    householder = np.eye(3) - 2.0 * p_n
    return {
        "P_normal": p_n,
        "P_tangent": p_t,
        "along_normal": p_n @ v,
        "in_tangent_plane": p_t @ v,
        "reflected": householder @ v,
        "cosine": float(v @ n / np.linalg.norm(v)),
    }


def test_projector_reference_properties():
    rng = np.random.default_rng(3)
    for k in (1, 2, 3):
        for _ in range(100):
            a = rng.standard_normal((3, k))
            p = projector_numpy(a)
            assert np.allclose(p @ p, p, atol=1e-12)
            assert np.allclose(p, p.T, atol=1e-12)
            assert np.isclose(np.trace(p), k, atol=1e-12)
            # P fixes the column space and kills its orthogonal complement
            assert np.allclose(p @ a, a, atol=1e-12)
            assert np.allclose(a.T @ (np.eye(3) - p), 0.0, atol=1e-12)


def test_projector_matches_qr_and_lstsq():
    """Three independent routes to the same projection: pinv, QR and least squares."""
    rng = np.random.default_rng(4)
    for _ in range(100):
        a = rng.standard_normal((3, 2))
        v = rng.standard_normal(3)
        q, _ = np.linalg.qr(a)
        coeff, *_ = np.linalg.lstsq(a, v, rcond=None)
        assert np.allclose(projector_numpy(a), q @ q.T, atol=1e-12)
        assert np.allclose(projector_numpy(a) @ v, a @ coeff, atol=1e-12)


def test_reflection_reference_properties():
    rng = np.random.default_rng(5)
    for _ in range(200):
        v = rng.standard_normal(3)
        n = rng.standard_normal(3)
        ref = project_onto_normal_numpy(v, n)
        nu = n / np.linalg.norm(n)
        assert np.isclose(np.linalg.norm(ref["reflected"]), np.linalg.norm(v), atol=1e-12)
        assert np.isclose(ref["reflected"] @ nu, -(v @ nu), atol=1e-12)
        assert np.allclose(ref["along_normal"] + ref["in_tangent_plane"], v, atol=1e-12)
        # the same reflection written the way the renderer does it: v - 2 (v . n) n
        assert np.allclose(ref["reflected"], v - 2.0 * (v @ nu) * nu, atol=1e-12)


def _load_projection_records():
    if not TRACE_PATH.exists():
        if os.environ.get("MFAD_REQUIRE_TRACE"):
            pytest.fail(f"{TRACE_PATH} missing; run test_projection_runner first")
        pytest.skip(f"{TRACE_PATH} not generated yet.")
    records = load_trace(TRACE_PATH)
    assert records, "trace has no records"
    assert all(r["stage"] == "projection" for r in records)
    return records


def _group(records, suffix_names):
    """Group records as {item: {suffix: record}} for names like 'normal_3/P_normal'."""
    groups: dict[str, dict] = {}
    for rec in records:
        item, _, suffix = rec["name"].rpartition("/")
        if suffix in suffix_names:
            groups.setdefault(item, {})[suffix] = rec
    return groups


def test_trace_checks_all_pass():
    for rec in _load_projection_records():
        for key, value in rec["checks"].items():
            if isinstance(value, bool):
                assert value is True, f"{rec['name']}: {key}"
            elif key.endswith("_error") or key == "residual_orthogonality":
                assert value < 1e-12, f"{rec['name']}: {key}={value}"


def test_trace_normal_projection_matches_numpy_reference():
    names = {"P_normal", "P_tangent", "along_normal", "in_tangent_plane", "reflected"}
    groups = {
        k: g
        for k, g in _group(_load_projection_records(), names).items()
        if k.startswith("normal_")
    }
    assert len(groups) >= 500
    for item, g in groups.items():
        checks = g["reflected"]["checks"]
        v = np.asarray(checks["input_v"], dtype=float)
        n = np.asarray(checks["input_normal"], dtype=float)
        ref = project_onto_normal_numpy(v, n)
        for key in names:
            assert np.allclose(g[key]["array"], ref[key], atol=1e-12, rtol=0.0), f"{item}/{key}"
        assert np.isclose(checks["cosine"], ref["cosine"], atol=1e-12)


def test_trace_subspace_projection_matches_numpy_reference():
    names = {"A", "P", "parallel", "perpendicular"}
    groups = {
        k: g
        for k, g in _group(_load_projection_records(), names).items()
        if not k.startswith("normal_")
    }
    assert {"line", "image_plane"} <= set(groups)
    assert len(groups) >= 200
    for item, g in groups.items():
        a = g["A"]["array"]
        v = np.asarray(g["P"]["checks"]["input_v"], dtype=float)
        p_ref = projector_numpy(a)
        assert np.allclose(g["P"]["array"], p_ref, atol=1e-12, rtol=0.0), item
        assert np.allclose(g["parallel"]["array"], p_ref @ v, atol=1e-12, rtol=0.0), item
        assert np.allclose(g["perpendicular"]["array"], v - p_ref @ v, atol=1e-12, rtol=0.0), item


def test_image_plane_projection_drops_depth():
    """The image plane A = [right, up] = [x, y] keeps x and y and drops z."""
    g = _group(_load_projection_records(), {"A", "P", "parallel", "perpendicular"})["image_plane"]
    v = np.asarray(g["P"]["checks"]["input_v"], dtype=float)
    assert np.allclose(g["parallel"]["array"], [v[0], v[1], 0.0], atol=1e-12)
    assert np.allclose(g["perpendicular"]["array"], [0.0, 0.0, v[2]], atol=1e-12)
