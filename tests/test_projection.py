import numpy as np


def reflect_numpy(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    n_unit = n / np.linalg.norm(n)
    return v - 2.0 * np.dot(v, n_unit) * n_unit


def diffuse_term_numpy(n: np.ndarray, l_dir: np.ndarray) -> float:
    n_unit = n / np.linalg.norm(n)
    l_unit = l_dir / np.linalg.norm(l_dir)
    return max(0.0, float(np.dot(n_unit, l_unit)))


def test_reflection_properties_numpy():
    """Verify reflection law preserves magnitude and angle."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        v = rng.standard_normal(3)
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)

        r = reflect_numpy(v, n)
        # Length preservation
        assert np.isclose(np.linalg.norm(r), np.linalg.norm(v), atol=1e-12)
        # Angle of incidence == angle of reflection
        assert np.isclose(np.dot(v, n) + np.dot(r, n), 0.0, atol=1e-12)


def test_diffuse_term_properties_numpy():
    """Verify diffuse term clamps to [0, 1]."""
    n = np.array([0.0, 1.0, 0.0])
    assert np.isclose(diffuse_term_numpy(n, np.array([0.0, 1.0, 0.0])), 1.0, atol=1e-12)
    assert np.isclose(diffuse_term_numpy(n, np.array([1.0, 0.0, 0.0])), 0.0, atol=1e-12)
    assert np.isclose(diffuse_term_numpy(n, np.array([0.0, -1.0, 0.0])), 0.0, atol=1e-12)
