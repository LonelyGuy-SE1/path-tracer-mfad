import numpy as np


def make_rotate_x_numpy(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def make_rotate_y_numpy(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def make_rotate_z_numpy(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def make_scale_numpy(s: np.ndarray) -> np.ndarray:
    return np.diag([s[0], s[1], s[2], 1.0])


def make_shear_numpy(s_xy: float = 0.0, s_xz: float = 0.0, s_yx: float = 0.0) -> np.ndarray:
    M = np.eye(4)
    M[0, 1] = s_xy
    M[0, 2] = s_xz
    M[1, 0] = s_yx
    return M


def test_rotations_isometry_numpy():
    """Verify rotations satisfy Q^T Q = I and det(Q) = 1."""
    rng = np.random.default_rng(1337)
    for _ in range(50):
        theta = rng.uniform(-np.pi, np.pi)
        for R_func in [make_rotate_x_numpy, make_rotate_y_numpy, make_rotate_z_numpy]:
            R = R_func(theta)
            Q = R[:3, :3]
            QtQ = Q.T @ Q
            assert np.allclose(QtQ, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-12)


def test_scale_shear_fail_isometry_numpy():
    """Verify scale and shear fail isometry (Issue #10)."""
    # Non-uniform scale
    S_nonuniform = make_scale_numpy(np.array([2.0, 1.0, 0.5]))
    Q_s = S_nonuniform[:3, :3]
    assert not np.allclose(Q_s.T @ Q_s, np.eye(3), atol=1e-6)

    # Uniform scale (det != 1)
    S_uniform = make_scale_numpy(np.array([2.0, 2.0, 2.0]))
    Q_su = S_uniform[:3, :3]
    assert not np.isclose(np.linalg.det(Q_su), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.det(Q_su), 8.0, atol=1e-12)

    # Shear
    Sh = make_shear_numpy(s_xy=1.5)
    Q_sh = Sh[:3, :3]
    assert not np.allclose(Q_sh.T @ Q_sh, np.eye(3), atol=1e-6)
