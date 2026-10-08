from __future__ import annotations

from pathlib import Path
import pytest
from viewer.view_trace import format_record, format_trace, view_trace_file
import numpy as np


def test_format_record():
    rec = {
        "stage": "test_stage",
        "name": "test_mat",
        "shape": [2, 2],
        "note": "Test matrix",
        "checks": {"is_valid": True, "error": 0.0},
        "array": np.array([[1.0, 0.0], [0.0, 1.0]]),
    }
    formatted = format_record(0, rec)
    assert "[000] Stage: test_stage" in formatted
    assert "Name: test_mat" in formatted
    assert "Shape: [2, 2]" in formatted
    assert "Note: Test matrix" in formatted
    assert "✓ is_valid: True" in formatted
    assert "• error: 0.0" in formatted
    assert "1." in formatted


def test_format_trace_empty():
    assert "empty" in format_trace([]).lower()


def test_view_trace_file(tmp_path: Path):
    from prep.trace import Trace

    trace = Trace()
    trace.record("stage_basis", "rot_identity", np.eye(3), note="Identity matrix", checks={"det_is_one": True})
    json_path = tmp_path / "test_trace.json"
    trace.save_json(json_path)

    out = view_trace_file(json_path)
    assert "MFAD Trace Viewer" in out
    assert "stage_basis" in out
    assert "rot_identity" in out
    assert "Identity matrix" in out
    assert "det_is_one: True" in out
