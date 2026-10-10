# projection

**Concept.** Orthogonal projection onto subspaces, normal hyperplanes, and lines in $\mathbb{R}^3$; decomposing arbitrary vectors into parallel and perpendicular components.

**Purpose.** Underpins core optics and shading interactions at every ray-object intersection:
1. **Specular Reflection (`stage_reflect`)**: Decomposes incident ray $v$ along unit normal $n$; inverts the normal projection component to reflect across the tangent plane:
   $$r = v - 2 (v \cdot n) n$$
2. **Diffuse Shading (`stage_diffuse_term`)**: Evaluates Lambert's cosine law by orthogonally projecting unit light direction $l$ onto surface normal $n$:
   $$f_{\text{diffuse}} = \max(0, n \cdot l)$$
3. **Shadow Ray Direction (`stage_shadow_direction`)**: Generates normalized direction vector and acne-free offset origin $p_{\text{offset}} = p + \epsilon l$.
4. **Camera Sensor Projection (`stage_perspective_project`)**: Projects 3D camera-space points onto the 2D film plane with perspective division $p_{xy} / p_z$.

**How.** Implemented in `stages/stage_projection.cpp`:
- Vectorised inner products ensure exact geometry preservation.
- Reflection validates conservation of vector magnitude and anti-symmetry of incidence vs reflection angles.
- Clamping in the diffuse term restricts projected energy to $[0, 1]$, enforcing thermodynamics and backface culling.

**Checks in the trace** (stage `projection`):
- `length_preserved` ($\|\|r\| - \|v\|\| < 10^{-12}$).
- `angle_preserved` ($|v \cdot n + r \cdot n| < 10^{-12}$).
- `in_range` ($0 \le \max(0, n \cdot l) \le 1$).
- `valid_dist` and `valid_dir` ($\|d\| = 1$ within numerical tolerance).

**Cost.** $< 5$ ns per hit; vector dot product and fused multiply-add executed in inner ray traversal.
