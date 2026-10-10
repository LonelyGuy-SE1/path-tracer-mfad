# basis

**Concept.** Orthonormal basis construction from a single vector; Gram-Schmidt orthogonalisation and branchless tangent space generation producing an orthogonal matrix $Q = [t, b, n] \in SO(3)$.

**Purpose.** Two vital roles in the rendering pipeline:
1. **Camera Frame (`stage_basis_camera`)**: Constructs the camera coordinate system (right $u$, up $v$, forward $w$) from eye position, look-at target, and user-specified up vector.
2. **Bounce Sampling Frame (`stage_basis_normal`)**: Builds a local tangent space $[t, b, n]$ at every ray-surface intersection so that cosine-weighted hemispherical rays sampled in local coordinates $(x, y, z)$ rotate back to world space via $v_{\text{world}} = Q v_{\text{local}}$.

**How.** Implemented in `stages/stage_basis.cpp`:
- **Duff et al. (2017) Formulation**: Branchless, continuous orthonormal basis from normal $n$:
  $$a = -\frac{1}{\operatorname{sign}(n_z) + n_z}, \quad f = n_x n_y a$$
  $$t = \begin{bmatrix} 1 + \operatorname{sign}(n_z) n_x^2 a \\ \operatorname{sign}(n_z) f \\ -\operatorname{sign}(n_z) n_x \end{bmatrix}, \quad b = \begin{bmatrix} f \\ \operatorname{sign}(n_z) + n_y^2 a \\ -n_y \end{bmatrix}$$
  Guarantees zero division by zero across the entire sphere, with smooth transitions around the nadir and zenith.
- **Gram-Schmidt Camera Basis**: Computes forward vector $w = \operatorname{normalize}(\text{eye} - \text{look\_at})$, orthogonal right vector $u = \operatorname{normalize}(\text{up} \times w)$, and bitangent $v = w \times u$. Includes singularity handling when view direction is collinear with up.

**Checks in the trace** (stage `basis`):
- `QtQ_is_identity` ($\|Q^T Q - I\|_\infty < 10^{-12}$).
- `det_is_one` ($|\det(Q) - 1| < 10^{-12}$).
- `roundtrip_is_identity` ($\|Q^T (Q v) - v\|_\infty < 10^{-12}$) verifying lossless isometric change-of-basis.

**Cost.** $\approx 15$ ns per hit for Duff basis; evaluated on every diffuse ray bounce across millions of Monte Carlo paths.
