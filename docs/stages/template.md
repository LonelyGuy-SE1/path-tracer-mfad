# <stage_name>

**Concept.** One sentence describing the core linear algebra concept, theorem, or factorization (e.g. orthogonal matrices, Gram-Schmidt orthonormalization, orthogonal projection, singular value decomposition).

**Purpose.** Where and why this linear algebra concept is required in the path tracer or geometry preprocessing pipeline.

**How.** Key mathematical formulations, algorithms, and implementation details:
1. Step-by-step mathematical equations and mappings to code.
2. Numerical stability, tolerances, and singular edge case handling.
3. Coordinate systems and transformation conventions.

**Checks in the trace.** Matrix and invariant checks recorded in the trace JSON (e.g. `QtQ_is_identity`, `det_is_one`, `error < 1e-12`, conservation laws).

**Cost.** Computational complexity, benchmark runtime on standard inputs, and invocation frequency (pre-render setup vs inner bounce loop).
