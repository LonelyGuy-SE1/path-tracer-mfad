# prep: Python preprocessing (mesh_checks + eigen)

```
python -m prep scenes/prep_demo.json -o scene_prepared.json --trace trace_prep.json
```

Runs `mesh_checks` then `eigen` on every mesh, and `eigen` on every quadric, and prints the time spent in each stage. Keys other than `meshes` and `quadrics` (camera, lights, materials, …) are copied through unchanged.

## Scene input

```json
{
  "meshes": [
    {"name": "cube", "file": "meshes/cube.obj", "material": "red", "transform": [[4x4]]},
    {"name": "tri", "vertices": [[x, y, z], ...], "faces": [[0, 1, 2], ...]}
  ],
  "quadrics": [
    {"name": "egg", "type": "ellipsoid", "center": [x, y, z], "radii": [a, b, c], "rotation_deg": [rx, ry, rz]},
    {"name": "ball", "type": "sphere", "center": [x, y, z], "radius": r},
    {"name": "raw", "matrix": [[4x4 symmetric]]}
  ]
}
```

Optional per mesh: `weld_tol` (default 1e-9), `rank_tol` (default 1e-8). Rotation order is `Rz Ry Rx`.

## Prepared output (what the C++ loader reads)

Per mesh:

- `vertices` (shape `vertices_shape` = `[3, N]`, columns), `faces` (`faces_shape` = `[M, 3]`), `normals` (`[3, M]`).
- Faces are reordered so each cluster is a contiguous range: `clusters[k] = {begin, end, center, axes, half_extents}`; faces `begin..end-1` belong to cluster `k`.
- `obb` – box for the whole mesh. `axes` is a 3x3 nested list, row major, **columns are the box axes**.
- `checks` – the mesh_checks counts.

Per quadric: `kind`, `center`, `axes` (columns), `radii`, `matrix`, `checks`.

Small arrays are inline JSON lists. Arrays with more than 4096 numbers are written to `<out>.bin` instead and the JSON holds `{"$bin": {"offset", "nbytes", "dtype"}}` (little endian `float64` or `int32`, row major in the listed shape). The trace uses the same scheme with `<trace>.bin`.
