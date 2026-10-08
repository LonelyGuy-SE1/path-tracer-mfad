#pragma once

#include "stage_interface.hpp"
#include "trace.hpp"

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mfad {

struct SvdDenoiseResult {
    Eigen::MatrixXd denoised;         // H x W low-rank approximation
    Eigen::VectorXd singular_values;  // full singular value spectrum
    int rank{0};                      // truncation rank used
    double frobenius_error{0.0};      // ||A - A_r||_F
    double relative_error{0.0};       // ||A - A_r||_F / ||A||_F
};

struct MultiChannelSvdResult {
    std::vector<Eigen::MatrixXd> denoised_channels;
    std::vector<Eigen::VectorXd> spectra;
    std::vector<double> relative_errors;
};

/**
 * @brief Computes truncated SVD approximation on a 2D matrix channel, returning low-rank
 * denoised matrix and singular value spectrum (Issue #18).
 *
 * @param channel Input H x W matrix.
 * @param rank Truncation target rank.
 * @param trace Optional trace recorder.
 * @param name Diagnostic identifier.
 */
StageResult<SvdDenoiseResult> stage_svd_denoise_channel(
    const Eigen::MatrixXd& channel, int rank, Trace* trace = nullptr,
    const std::string& name = "svd_denoise_channel");

/**
 * @brief Performs per-channel truncated SVD across multiple image channels (e.g. RGB).
 */
StageResult<MultiChannelSvdResult> stage_svd_denoise_image(
    const std::vector<Eigen::MatrixXd>& channels, int rank, Trace* trace = nullptr,
    const std::string& name = "svd_denoise_image");

}  // namespace mfad
