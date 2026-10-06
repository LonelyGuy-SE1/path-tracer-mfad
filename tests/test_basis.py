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
