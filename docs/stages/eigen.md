# eigen

**Concept.** Eigenvalues, eigenvectors, symmetric diagonalisation `A = R Λ Rᵀ` with `R` orthogonal.

**Purpose.** Two places in the renderer:

1. **PCA oriented bounding boxes** for mesh clusters that feed the BVH.
2. **Quadric axes** for ellipsoid / sphere intersection.

**PCA boxes** (`stage_eigen_obb`). Covariance of the cluster is area weighted over triangles by default (Gottschalk's continuous formula), so dense tessellation on one side doesn't tilt the box; `weighting="points"` gives plain vertex covariance. `eigh` → eigenvalues sorted descending, eigenvectors as columns, sign fixed, `det = +1`. Points are projected onto the axes to get extents, centre and half extents. When eigenvalues are (near) equal, e.g. a cube, the eigenvectors are arbitrary and the box can be loose; the stage then also tries world axes and frames from the largest face normals and keeps the smallest volume. The PCA frame is always logged; `box_frame` says which frame was used.

`cluster_faces` builds the clusters top down: split face centroids at the median along their principal axis until a leaf has ≤ `max_faces` faces.

**Quadrics** (`stage_eigen_quadric`). Input is a symmetric 4x4 `Q` with `xᵀQx = 0`. With `A` the upper 3x3 block and `b` the last column: centre `c = -A⁻¹b`, `s = cᵀAc - d`, diagonalise `A`, semi axes `rᵢ = sqrt(s / λᵢ)`. Classified as `ellipsoid`, `sphere`, `hyperboloid`, `empty`, `point`, or `degenerate` (singular `A`). The renderer intersects in unit sphere space: `o' = diag(1/r) Rᵀ (o - c)`, `d' = diag(1/r) Rᵀ d`, then solves the unit sphere quadratic; `t` is the same in both spaces. `intersect_ellipsoid` is the NumPy reference for that.

**Checks in the trace** (stage `eigen`): `QtQ_is_identity`, `det_is_one`, `eig_residual_ok` (‖AR − RΛ‖), `diag_reconstruction_ok`, `all_points_inside`, `eigengap_min`, `obb_over_aabb`; for quadrics also `rebuild_ok` and `axis_tips_on_surface`.

**Cost.** ~3 s for 200k faces into 256 face clusters, run once before rendering.
