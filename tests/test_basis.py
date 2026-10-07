from pathlib import Path
from prep.trace import load_trace

TRACE_PATH = Path("trace_basis.json")

def gram_schmidt_numpy(normal: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """NumPy reference for stage_basis_gram_schmidt.
    The tangent is the guide projected onto the plane orthogonal to n, written here with the
    projector (I - n n^T) so it does not share code with the C++ subtraction form.
    """
    n = normal / np.linalg.norm(normal)
    proj_plane = np.eye(3) - np.outer(n, n)
    u = proj_plane @ guide
    if u @ u < 1e-12:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(n[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        u = proj_plane @ fallback
    t = u / np.linalg.norm(u)
    b = np.cross(n, t)
    return np.column_stack([t, b, n])

def test_gram_schmidt_reference_properties():
    rng = np.random.default_rng(7)
    guide = np.array([0.0, 1.0, 0.0])
    for _ in range(200):
        n = rng.standard_normal(3)
        Q = gram_schmidt_numpy(n, guide)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-12)
        assert np.allclose(Q[:, 2], n / np.linalg.norm(n), atol=1e-12)
    # guide parallel to the normal takes the fallback axis and must still be a valid frame
    Q = gram_schmidt_numpy(np.array([0.0, 2.0, 0.0]), guide)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-12)


def _load_basis_records():
    if not TRACE_PATH.exists():
        if os.environ.get("MFAD_REQUIRE_TRACE"):
            pytest.fail(f"{TRACE_PATH} missing; run test_basis_runner first")
        pytest.skip(f"{TRACE_PATH} not generated yet.")
    records = load_trace(TRACE_PATH)
    assert records, "trace has no records"
    return records


def test_trace_basis_checks():
    """Every logged frame carries passing invariant checks."""
    for rec in _load_basis_records():


def test_trace_duff_matches_numpy_reference():
    """C++ Duff frames equal the NumPy reference for the same input normal."""
    records = [r for r in _load_basis_records() if r["name"].startswith("duff_sample_")]
    assert len(records) >= 1000
    worst = 0.0
    for rec in records:
        q = rec["array"]
        assert q.shape == (3, 3)
        ref = duff_basis_numpy(np.asarray(rec["checks"]["input_normal"], dtype=float))
        worst = max(worst, float(np.abs(q - ref).max()))
        assert np.allclose(q, ref, atol=1e-12, rtol=0.0), rec["name"]
    assert worst < 1e-12


def test_trace_gram_schmidt_matches_numpy_reference():
    """C++ Gram-Schmidt frames equal the NumPy reference for the same normal and guide."""
    records = [r for r in _load_basis_records() if r["name"].startswith("gs_sample_")]
    assert len(records) >= 1000
    for rec in records:
        ref = gram_schmidt_numpy(
            np.asarray(rec["checks"]["input_normal"], dtype=float),
            np.asarray(rec["checks"]["input_guide"], dtype=float),
        )
        assert np.allclose(rec["array"], ref, atol=1e-12, rtol=0.0), rec["name"]


def test_trace_recorded_matrices_are_rotations():
    """Recomputed from the logged matrices alone: Q^T Q = I, det = +1, T x B = N."""
    for rec in _load_basis_records():
        q = rec["array"]
        assert np.allclose(q.T @ q, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(q), 1.0, atol=1e-12)
        assert np.allclose(np.cross(q[:, 0], q[:, 1]), q[:, 2], atol=1e-12)
