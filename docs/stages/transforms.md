# transforms

**Concept.** Matrix representation of affine transforms in $\mathbb{P}^3$ homogeneous coordinates; the special orthogonal group $SO(3)$ of rigid 3D rotations characterised by $Q^T Q = I$ and $\det(Q) = +1$.

**Purpose.** Position, orient, and scale meshes, quadrics, and the camera in world space during scene loading:
1. **Points**: $P' = M \begin{bmatrix} P \\ 1 \end{bmatrix}$ with dehomogenisation $P'/w$.
2. **Direction Vectors**: $v' = M_{3\times 3} v$.
3. **Surface Normals**: $n' = \operatorname{normalize}\left((M_{3\times 3}^{-1})^T n\right)$ preserving tangent-normal orthogonality under non-uniform scaling.

**How.** Implemented in `stages/stage_transforms.cpp`:
- **Elementary Rotations**: `make_rotate_x`, `make_rotate_y`, `make_rotate_z` evaluate trigonometric sub-blocks preserving $SO(3)$ structure.
- **Axis-Angle Rotation**: `make_rotate_axis(a, \theta)` constructs Rodrigues' rotation matrix for arbitrary unit axis $a$.
- **Translations & Scale**: `make_translate(t)` sets the fourth column; `make_scale(s)` forms a diagonal scaling matrix.
- **Shear**: `make_shear` injects off-diagonal coupling, distorting angles while preserving volume when unit-triangular.
- **Composition**: Left-multiplication sequence $M = T_k \cdots T_2 T_1$.

**Checks in the trace** (stage `transforms`):
- `QtQ_error` ($\|Q^T Q - I\|_\infty$) and `QtQ_is_identity` ($< 10^{-12}$).
- `det` and `det_is_one` ($|\det(Q) - 1| < 10^{-12}$).
- `passes_isometry`: `true` for rigid rotations (distance and angle preserving); `false` for uniform scale ($\det = s^3 \neq 1$), non-uniform scale ($Q^T Q \neq I$), and shear.

**Cost.** $\approx 0.05$ ms per object during scene ingestion; evaluated once before ray generation.
