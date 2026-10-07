# mesh_checks

**Concept.** Rank, linear independence, basis selection. A triangle `(v0, v1, v2)` is only a real surface if its edge matrix `E = [v1 - v0, v2 - v0]` (3x2) has rank 2.

**Purpose.** Throw away triangles that cost intersection work but can never contribute a valid hit or normal, before anything reaches the BVH.

**How.** Fully vectorised NumPy over all faces at once (`prep/stage_mesh_checks.py`):

1. `bad_index` – index outside `[0, N)`.
2. `nonfinite_vertex` – NaN/inf coordinates.
3. `repeated_index` – same index twice in a face.
4. Vertices are welded on a `weld_tol` grid; `coincident_vertices` – distinct indices that land on the same point.
5. Singular values of `E` from the closed form of the 2x2 Gram matrix `EᵀE` (σ_max² = λ_max, σ_min² = |e1×e2|²/λ_max).
   `rank0_collapsed` when σ_max ≤ `abs_tol`, `rank1_collinear` when σ_min/σ_max ≤ `rel_tol` (scale invariant, so huge slivers are caught and tiny valid triangles are kept).
6. `duplicate_face` – same vertex set as an earlier face (either winding).
7. Unreferenced vertices are dropped and indices are compacted.

Mesh level: affine rank from the singular values of the centred vertex scatter (`mesh_is_flat` when < 3), and a greedy affine basis (farthest point, farthest from the line, farthest from the plane).

**Outcome.** Clean `3xN` vertices, `Mx3` faces, unit normals, areas, per face drop reason, and an old→new vertex map. Trace stage `mesh_checks` logs face singular values, ranks, drop reasons and counts (`all_kept_faces_rank2`, `normals_are_unit`, `dropped_*`).

**Cost.** ~1.7 s for 400k faces / 200k vertices, run once before rendering.
