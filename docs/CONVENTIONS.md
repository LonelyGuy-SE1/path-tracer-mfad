# Project Conventions

This document specifies the mathematical, geometric, and software engineering conventions adhered to throughout the MFAD Linear Algebra Path Tracer codebase.

---

## 1. Coordinate Systems & Geometry

* **World Coordinate Frame**: Standard right-handed 3D Cartesian coordinates.
  * $+X$-axis: points **right**.
  * $+Y$-axis: points **up** (Zenith).
  * $+Z$-axis: points **out of the screen** toward the camera viewer (Nadir / backwards).
  * The camera looks down the negative $Z$-axis ($-Z$) in default view space.
* **Vector Representations**:
  * All vectors are **column vectors** ($v \in \mathbb{R}^{3 \times 1}$ or $\mathbb{R}^{4 \times 1}$).
  * Matrix multiplication follows standard standard linear algebra order: $v' = M v$.

---

## 2. Numerical Precision

* **Linear Algebra Stages (`stages/`)**:
  * **Double precision** (`double`, `Eigen::Vector3d`, `Eigen::Matrix3d`, `Eigen::MatrixXd`).
  * Used for orthonormal basis constructions, projections, PCA/eigen decompositions, least-squares pose estimation, and SVD denoising.
  * Ensures strict numerical verification: $\|Q^T Q - I_3\|_\infty < 10^{-12}$ and $|\det(Q) - 1.0| < 10^{-12}$.
* **Rendering Core (`core/`)**:
  * **Single precision** (`float`, `Eigen::Vector3f`).
  * Used for ray tracing loops, bounding volume hierarchy traversals, intersection tests, and material scattering to optimize cache footprint and SIMD throughput.
  * Vectors cast cleanly across boundaries via `.cast<double>()` and `.cast<float>()`.

---

## 3. Screen, Camera, and Image Buffers

* **Viewport Normalized Coordinates $(u, v)$**:
  * Origin $(0, 0)$ is at the **bottom-left** corner of the viewport.
  * Top-right is $(1, 1)$.
  * $u \in [0, 1]$ increases from left to right.
  * $v \in [0, 1]$ increases from bottom to top.
* **Raster Buffer Coordinates $(x, y)$**:
  * In memory and image file formats (e.g., PNG via `stb_image_write`), row $y = 0$ is the **top scanline**, and row $y = \text{height} - 1$ is the **bottom scanline**.
  * Column $x = 0$ is the left edge, and $x = \text{width} - 1$ is the right edge.
* **Scanline to Viewport Mapping**:
  * Both gradient generation and scene ray tracing flip the vertical raster index:
    $$u = \frac{x + \Delta x}{\text{width} - 1}$$
    $$v = \frac{\text{height} - 1 - y + \Delta y}{\text{height} - 1}$$
    where $\Delta x, \Delta y \in [0, 1)$ are sub-pixel stratified random jitter offsets.
* **Color Space and Gamma Correction**:
  * Internal ray tracing and radiance calculations are strictly **linear RGB** ($[0, \infty)$ HDR).
  * The `ImageBuffer::write_png` pipeline applies standard gamma $2.2$ correction before quantizing to 8-bit unsigned integers:
    $$C_{\text{sRGB}} = \mathrm{clamp}(C_{\text{linear}}, 0.0, 1.0)^{1 / 2.2}$$
    $$I_{\text{8bit}} = \lfloor 255.999 \times C_{\text{sRGB}} \rfloor$$

---

## 4. Orthonormal Bases & Matrix Representations

* **Orthonormal Basis Matrix**:
  * A local coordinate frame at a surface normal $n$ is defined by column vectors in $\mathrm{SO}(3)$:
    $$Q = \begin{bmatrix} T & B & N \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$
    where $T$ is the tangent, $B$ is the bitangent, and $N$ is the surface unit normal.
  * Column vectors form a right-handed basis: $T \times B = N$, with $\det(Q) = +1$ and $Q^T Q = I_3$.
* **Coordinate Transformations**:
  * **Local to World**: $v_{\text{world}} = Q \, v_{\text{local}}$
  * **World to Local**: $v_{\text{local}} = Q^{-1} v_{\text{world}} = Q^T v_{\text{world}}$

---

## 5. Architectural Separation & Naming

* **`stages/` Directory**:
  * Files are named `stage_<name>.hpp` and `stage_<name>.cpp`.
  * Pure linear algebra functions operating on Eigen types.
  * Must **never** include headers from `core/` (no renderer or scene types).
  * Optionally logs matrix operations and invariants to `mfad::Trace`.
* **`core/` Directory**:
  * Contains rendering primitives (`Ray`, `Camera`, `ImageBuffer`, `Sphere`, `Plane`, `Material`, `BVH`, etc.).
  * Feeds geometric data into `stages/` to compute linear algebra transforms.
* **`trace/` Directory**:
  * Trace logger emitting structured JSON records with matrix shape, contents, and invariant flags.
* **`tests/` Directory**:
  * C++ unit tests in `test_<name>.cpp`.
  * Corresponding Python validation tests in `test_<name>.py` comparing C++ trace outputs against pure NumPy references.
