#include "stage_svd_denoise.hpp"

#include <algorithm>
#include <cmath>

namespace mfad {

StageResult<SvdDenoiseResult> stage_svd_denoise_channel(
    const Eigen::MatrixXd& channel, int rank, Trace* trace,
    const std::string& name) {
    StageResult<SvdDenoiseResult> result;
    result.stage_name = "stage_svd_denoise";

    Eigen::Index rows = channel.rows();
    Eigen::Index cols = channel.cols();
    if (rows == 0 || cols == 0) {
        result.success = false;
        result.note = "Input channel matrix is empty";
        return result;
    }

    int max_rank = static_cast<int>(std::min(rows, cols));
    int target_rank = std::clamp(rank, 1, max_rank);

    // Compute thin SVD: A = U * S * V^T
    Eigen::BDCSVD<Eigen::MatrixXd> svd(channel, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::MatrixXd& U = svd.matrixU();
    const Eigen::VectorXd& S = svd.singularValues();
    const Eigen::MatrixXd& V = svd.matrixV();

    // Reconstruct truncated rank-r approximation: A_r = U[:, :r] * diag(S[:r]) * V[:, :r]^T
    Eigen::MatrixXd U_r = U.leftCols(target_rank);
    Eigen::VectorXd S_r = S.head(target_rank);
    Eigen::MatrixXd V_r = V.leftCols(target_rank);

    Eigen::MatrixXd A_r = U_r * S_r.asDiagonal() * V_r.transpose();

    double orig_norm = channel.norm();
    double error = (channel - A_r).norm();
    double rel_error = (orig_norm > 1e-12) ? (error / orig_norm) : 0.0;

    SvdDenoiseResult res;
    res.denoised = std::move(A_r);
    res.singular_values = S;
    res.rank = target_rank;
    res.frobenius_error = error;
    res.relative_error = rel_error;

    if (trace != nullptr) {
        std::map<std::string, nlohmann::json> checks;
        checks["rank"] = target_rank;
        checks["max_rank"] = max_rank;
        checks["frobenius_error"] = error;
        checks["relative_error"] = rel_error;
        checks["energy_captured"] = (orig_norm > 1e-12) ? (1.0 - rel_error * rel_error) : 1.0;

        trace->record_matrix(result.stage_name, name + "_denoised", res.denoised,
                             "Rank-" + std::to_string(target_rank) + " SVD approximation", checks);
        trace->record_matrix(result.stage_name, name + "_singular_values", res.singular_values,
                             "Singular value spectrum", checks);
    }

    result.value = std::move(res);
    result.success = true;
    return result;
}

StageResult<MultiChannelSvdResult> stage_svd_denoise_image(
    const std::vector<Eigen::MatrixXd>& channels, int rank, Trace* trace,
    const std::string& name) {
    StageResult<MultiChannelSvdResult> result;
    result.stage_name = "stage_svd_denoise";

    if (channels.empty()) {
        result.success = false;
        result.note = "No channels provided";
        return result;
    }

    MultiChannelSvdResult multi_res;
    for (size_t c = 0; c < channels.size(); ++c) {
        std::string ch_name = name + "_ch" + std::to_string(c);
        auto ch_res = stage_svd_denoise_channel(channels[c], rank, trace, ch_name);
        if (!ch_res.success) {
            result.success = false;
            result.note = ch_res.note;
            return result;
        }
        multi_res.denoised_channels.push_back(ch_res.value.denoised);
        multi_res.spectra.push_back(ch_res.value.singular_values);
        multi_res.relative_errors.push_back(ch_res.value.relative_error);
    }

    result.value = std::move(multi_res);
    result.success = true;
    return result;
}

}  // namespace mfad
