from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from prep.trace import load_trace
except ImportError:
    # Allow running directly as a standalone script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from prep.trace import load_trace


def format_record(idx: int, rec: dict[str, Any], precision: int = 4, max_elements: int = 16) -> str:
    """Format a single trace record into a readable human-friendly string."""
    stage = rec.get("stage", "unknown_stage")
    name = rec.get("name", "unnamed")
    shape = rec.get("shape", [])
    note = rec.get("note", "")
    checks = rec.get("checks", {})
    array = rec.get("array")

    lines: list[str] = []
    lines.append(f"[{idx:03d}] Stage: {stage} | Name: {name} | Shape: {shape}")

    if note:
        lines.append(f"      Note: {note}")

    if checks:
        lines.append("      Checks:")
        for check_name, check_val in checks.items():
            status = "✓" if check_val is True else ("✗" if check_val is False else "•")
            lines.append(f"        {status} {check_name}: {check_val}")

    if array is not None:
        lines.append("      Matrix:")
        arr = np.asarray(array)
        with np.printoptions(
            precision=precision, suppress=True, edgeitems=3, threshold=max_elements
        ):
            arr_str = str(arr)
            for line in arr_str.splitlines():
                lines.append(f"        {line}")

    return "\n".join(lines)


def format_trace(records: list[dict[str, Any]], precision: int = 4) -> str:
    """Walk the trace in stage order and format every stage, matrix, note, and check."""
    if not records:
        return "Trace is empty (0 records)."

    out: list[str] = []
    out.append("=" * 70)
    out.append(f"  MFAD Trace Viewer (Total Records: {len(records)})")
    out.append("=" * 70)

    for i, rec in enumerate(records):
        out.append(format_record(i, rec, precision=precision))
        out.append("-" * 70)

    return "\n".join(out)


def view_trace_file(path: str | Path, precision: int = 4) -> str:
    """Load and format a trace file from disk."""
    records = load_trace(path)
    return format_trace(records, precision=precision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Viewer v0: Walks the trace in stage order and prints each matrix, note, and check (Issue #7)."
    )
    parser.add_argument("trace_file", type=str, help="Path to the .json trace file")
    parser.add_argument("--precision", type=int, default=4, help="Floating point display precision")
    args = parser.parse_args(argv)

    path = Path(args.trace_file)
    if not path.exists():
        print(f"[ERROR] Trace file not found: {path}", file=sys.stderr)
        return 1

    try:
        output = view_trace_file(path, precision=args.precision)
        print(output)
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to read trace: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
