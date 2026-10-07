import argparse
import sys

from .scene import prepare_file


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m prep")
    p.add_argument("scene")
    p.add_argument("-o", "--out", default="scene_prepared.json")
    p.add_argument("--trace", default="trace_prep.json")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--cluster-faces", type=int, default=64)
    a = p.parse_args()
    timings = prepare_file(a.scene, a.out, None if a.no_trace else a.trace, a.cluster_faces)
    for k, v in timings.items():
        print(f"{k}: {v * 1000:.1f} ms")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
