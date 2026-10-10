#include "stage_svd_denoise.hpp"
#include "trace.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_svd_denoise_synthetic_rank() {
    std::cout << "  Testing synthetic rank-2 matrix denoising..." << std::endl;
    const int H = 20;
    const int W = 25;

    // Generate rank-2 signal
    Eigen::VectorXd u1 = Eigen::VectorXd::LinSpaced(H, 1.0, 2.0);
    Eigen::VectorXd v1 = Eigen::VectorXd::LinSpaced(W, 0.5, 1.5);
    Eigen::VectorXd u2 = Eigen::VectorXd::LinSpaced(H, -1.0, 1.0);
    Eigen::VectorXd v2 = Eigen::VectorXd::LinSpaced(W, 2.0, -1.0);

    Eigen::MatrixXd signal = u1 * v1.transpose() + u2 * v2.transpose();

    // Add mild noise
    Eigen::MatrixXd noise = 0.05 * Eigen::MatrixXd::Random(H, W);
    Eigen::MatrixXd noisy_image = signal + noise;

    mfad::Trace trace;
    auto res = mfad::stage_svd_denoise_channel(noisy_image, 2, &trace, "synth_rank2");
    assert(res.success);
    assert(res.value.rank == 2);
    assert(res.value.denoised.rows() == H);
    assert(res.value.denoised.cols() == W);
    assert(res.value.singular_values.size() == std::min(H, W));

    // Singular values must be sorted non-increasing
    for (int i = 0; i < res.value.singular_values.size() - 1; ++i) {
        assert(res.value.singular_values(i) >= res.value.singular_values(i + 1));
    }

    // Denoised image should be closer to true signal than noisy image
    double noisy_err = (noisy_image - signal).norm();
    double denoised_err = (res.value.denoised - signal).norm();
    assert(denoised_err < noisy_err);
}

void test_svd_rank_sweep_monotonic_error() {
    std::cout << "  Testing rank sweep monotonic error decrease..." << std::endl;
    const int N = 15;
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(N, N);

    double prev_error = 1e9;
    for (int r = 1; r <= N; ++r) {
        auto res = mfad::stage_svd_denoise_channel(mat, r);
        assert(res.success);
        assert(res.value.frobenius_error <= prev_error + 1e-10);
        prev_error = res.value.frobenius_error;
    }
    // Full rank approximation must have zero error
    auto full_res = mfad::stage_svd_denoise_channel(mat, N);
    assert(full_res.value.frobenius_error < 1e-10);
}

void test_multichannel_svd_denoise() {
    std::cout << "  Testing multi-channel RGB image denoising..." << std::endl;
    const int H = 16;
    const int W = 16;
    std::vector<Eigen::MatrixXd> channels = {Eigen::MatrixXd::Random(H, W),
                                             Eigen::MatrixXd::Random(H, W),
                                             Eigen::MatrixXd::Random(H, W)};

    auto res = mfad::stage_svd_denoise_image(channels, 4);
    assert(res.success);
    assert(res.value.denoised_channels.size() == 3);
    assert(res.value.spectra.size() == 3);
    assert(res.value.relative_errors.size() == 3);
}

int main() {
    std::cout << "[TEST] Running stage_svd_denoise tests (Issue #18)..." << std::endl;
    test_svd_denoise_synthetic_rank();
    test_svd_rank_sweep_monotonic_error();
    test_multichannel_svd_denoise();
    std::cout << "[PASS] All stage_svd_denoise tests passed!" << std::endl;
    return 0;
}
