import json
import os

import numpy as np
import pytest


def duff_basis_numpy(normal: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of Duff et al. (2017) orthonormal basis."""
    n = normal / np.linalg.norm(normal)
    sign = np.copysign(1.0, n[2])
    a = -1.0 / (sign + n[2])
    f = n[0] * n[1] * a
    t = np.array([1.0 + sign * n[0] * n[0] * a, sign * f, -sign * n[0]])
    b = np.array([f, sign + n[1] * n[1] * a, -n[1]])
    return np.column_stack([t, b, n])


def test_duff_basis_properties():
    """Verify orthonormality and right-handedness over unit sphere."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        v = rng.standard_normal(3)
        n = v / np.linalg.norm(v)
        Q = duff_basis_numpy(n)

        # 1. Orthogonality: Q^T * Q = I
        QtQ = Q.T @ Q
        assert np.allclose(QtQ, np.eye(3), atol=1e-12)

        # 2. Right-handed: det(Q) = 1
        det = np.linalg.det(Q)
        assert np.isclose(det, 1.0, atol=1e-12)

        # 3. Third column is the normal
        assert np.allclose(Q[:, 2], n, atol=1e-12)

        # 4. Round-trip coordinate transformation
        v_local = rng.standard_normal(3)
        v_world = Q @ v_local
        v_rec = Q.T @ v_world
        assert np.allclose(v_rec, v_local, atol=1e-12)


def test_trace_basis_json():
    """Verify generated trace JSON checks if file exists."""
    trace_path = "trace_basis.json"
    if not os.path.exists(trace_path):
        pytest.skip(f"{trace_path} not generated yet.")

    with open(trace_path) as f:
        records = json.load(f)

    for rec in records:
        assert rec["stage"] == "basis"
        checks = rec["checks"]
        assert checks["QtQ_is_identity"] is True
        assert checks["det_is_one"] is True
        assert checks["roundtrip_is_identity"] is True
        assert checks["QtQ_error"] < 1e-12
        assert checks["roundtrip_error"] < 1e-12


def camera_basis_numpy(look_from: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of camera Gram-Schmidt orthonormal basis."""
    view = look_from - look_at
    if np.linalg.norm(view) < 1e-12:
        view = np.array([0.0, 0.0, 1.0])
    w = view / np.linalg.norm(view)
    u_cross = np.cross(up, w)
    if np.linalg.norm(u_cross) < 1e-12:
        alt_up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_cross = np.cross(alt_up, w)
    u = u_cross / np.linalg.norm(u_cross)
    v = np.cross(w, u)
    return np.column_stack([u, v, w])


def test_camera_basis_properties():
    """Verify camera frame orthonormality and singularity fallback."""
    Q = camera_basis_numpy(
        np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-12)

    # Test singularity: look parallel to up
    Q_sing = camera_basis_numpy(
        np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert np.allclose(Q_sing.T @ Q_sing, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(Q_sing), 1.0, atol=1e-12)

    # Random cameras
    rng = np.random.default_rng(123)
    for _ in range(50):
        eye = rng.standard_normal(3)
        target = rng.standard_normal(3)
        up_vec = np.array([0.0, 1.0, 0.0])
        Q_rand = camera_basis_numpy(eye, target, up_vec)
        assert np.allclose(Q_rand.T @ Q_rand, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(Q_rand), 1.0, atol=1e-12)
