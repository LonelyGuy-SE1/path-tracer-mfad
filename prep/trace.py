from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INLINE_LIMIT = 4096


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class BinStore:
    def __init__(self, inline_limit: int = INLINE_LIMIT) -> None:
        self.inline_limit = inline_limit
        self.chunks: list[bytes] = []
        self.offset = 0

    def pack(self, array: np.ndarray) -> Any:
        array = np.ascontiguousarray(array)
        if array.size <= self.inline_limit:
            return array.tolist()
        if array.dtype.kind == "f":
            array = array.astype("<f8")
            dtype = "float64"
        else:
            array = array.astype("<i4")
            dtype = "int32"
        raw = array.tobytes()
        ref = {"$bin": {"offset": self.offset, "nbytes": len(raw), "dtype": dtype}}
        self.chunks.append(raw)
        self.offset += len(raw)
        return ref

    def write(self, path: Path) -> bool:
        if not self.chunks:
            return False
        with open(path, "wb") as f:
            for chunk in self.chunks:
                f.write(chunk)
        return True


def unpack(value: Any, shape: list[int], blob: bytes | None) -> np.ndarray:
    if isinstance(value, dict) and "$bin" in value:
        if blob is None:
            raise ValueError("binary sidecar missing")
        ref = value["$bin"]
        dtype = "<f8" if ref["dtype"] == "float64" else "<i4"
        raw = blob[ref["offset"] : ref["offset"] + ref["nbytes"]]
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    return np.asarray(value).reshape(shape)


class Trace:
    def __init__(self, inline_limit: int = INLINE_LIMIT) -> None:
        self.records: list[dict[str, Any]] = []
        self.store = BinStore(inline_limit)

    def record(
        self,
        stage: str,
        name: str,
        data: Any,
        note: str = "",
        checks: dict[str, Any] | None = None,
    ) -> None:
        array = np.asarray(data)
        self.records.append(
            {
                "stage": stage,
                "name": name,
                "shape": list(array.shape),
                "data": self.store.pack(array),
                "note": note,
                "checks": to_jsonable(checks or {}),
            }
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        bin_path = path.with_suffix(".bin")
        root = self.records
        if self.store.write(bin_path):
            for rec in root:
                if isinstance(rec["data"], dict):
                    rec["data"]["$bin"]["file"] = bin_path.name
        path.write_text(json.dumps(root, indent=2))
        return path


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = json.loads(path.read_text())
    bin_path = path.with_suffix(".bin")
    blob = bin_path.read_bytes() if bin_path.exists() else None
    for rec in records:
        rec["array"] = unpack(rec["data"], rec["shape"], blob)
    return records
