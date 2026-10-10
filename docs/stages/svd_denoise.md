# svd_denoise

**Concept.** Singular Value Decomposition (SVD) and the Eckart-Young-Mirsky theorem for optimal low-rank matrix approximation. Any image matrix $A \in \mathbb{R}^{m \times n}$ factorises as $A = U \Sigma V^T = \sum_{i=1}^{\min(m, n)} \sigma_i u_i v_i^T$. Truncating to rank $r$ yields the closest rank-$r$ matrix in Frobenius norm:
$$A_r = \sum_{i=1}^r \sigma_i u_i v_i^T, \quad \|A - A_r\|_F = \sqrt{\sum_{i=r+1}^{\min(m, n)} \sigma_i^2}$$

**Purpose.** Denoises raw Monte Carlo path-traced renders (`final_demo_pathtraced.png`). Coherent scene geometry (smooth gradients, walls, spherical boundaries) concentrates in the dominant singular modes, whereas high-frequency stochastic Monte Carlo noise (fireflies, variance) spreads across the lower singular spectrum. Truncating the tail filters noise while preserving structural contours.

**How.** Implemented in `stages/stage_svd_denoise.cpp`:
1. Separates the rendered image buffer into three independent $H \times W$ matrices (Red, Green, Blue channels), where $H = 360, W = 640$.
2. Computes thin SVD via Eigen's divide-and-conquer solver `Eigen::BDCSVD<Eigen::MatrixXd>`:
   $$A = U_{m \times k} \operatorname{diag}(\sigma_1, \ldots, \sigma_k) V_{n \times k}^T \quad (k = \min(m, n) = 360)$$
3. Reconstructs rank-$r$ channel approximation using $r = 120$:
   $$A_{120} = U_{[:, :120]} \operatorname{diag}(\sigma_{1:120}) V_{[:, :120]}^T$$
4. Clamps pixel values to $\ge 0$ to eliminate negative ringing artifacts, producing `final_demo_denoised.png`.

**Checks in the trace** (stage `svd_denoise`):
- `rank` ($r = 120$) and `max_rank` ($360$).
- `frobenius_error` ($\|A - A_r\|_F$) and `relative_error` ($\|A - A_r\|_F / \|A\|_F$).
- `energy_captured`: $\sum_{i=1}^r \sigma_i^2 / \sum_{i=1}^k \sigma_i^2$ (typically $> 99.4\%$).
- Full singular value decay spectrum logged for compression analysis.

**Cost.** $\approx 0.2$ s across all three RGB channels for a $640 \times 360$ image; executed once as the final pipeline post-process.
