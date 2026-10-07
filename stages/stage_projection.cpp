#include "stage_projection.hpp"

#include "trace.hpp"

#include <cmath>
#include <stdexcept>

namespace mfad {

ProjectionResult stage_projection_subspace(const Eigen::MatrixXd& A, const Eigen::Vector3d& v,
                                           Trace* trace, const std::string& item_name) {
    if (A.rows() != 3 || A.cols() < 1 || A.cols() > 3) {
        throw std::invalid_argument("stage_projection_subspace: A must be 3 x k with k in 1..3");
    }

    // P = A (A^T A)^-1 A^T. Solve against A^T instead of forming the inverse. The QR is
    // rank revealing so a dependent set of columns is rejected instead of giving garbage.
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    qr.setThreshold(1e-12);
    if (qr.rank() != A.cols()) {
        throw std::invalid_argument("stage_projection_subspace: columns of A are dependent");
    }
    const Eigen::MatrixXd gram = A.transpose() * A;
    const Eigen::MatrixXd solved = gram.ldlt().solve(A.transpose());

    ProjectionResult res;
    res.P = A * solved;
    res.parallel = res.P * v;
    res.perpendicular = v - res.parallel;

    if (trace != nullptr) {
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        const double idempotent_error = (res.P * res.P - res.P).cwiseAbs().maxCoeff();
        const double symmetric_error = (res.P - res.P.transpose()).cwiseAbs().maxCoeff();
        const double trace_error = std::abs(res.P.trace() - static_cast<double>(A.cols()));
        const double residual_orthogonality =
            (A.transpose() * res.perpendicular).cwiseAbs().maxCoeff();
        const double pythagoras_error = std::abs(v.squaredNorm() - res.parallel.squaredNorm() -
                                                 res.perpendicular.squaredNorm());
        const double complement_error =
            ((I - res.P) * (I - res.P) - (I - res.P)).cwiseAbs().maxCoeff();

        const std::map<std::string, nlohmann::json> checks = {
            {"k", static_cast<int>(A.cols())},
            {"idempotent_error", idempotent_error},
            {"P2_is_P", idempotent_error < 1e-12},
            {"symmetric_error", symmetric_error},
            {"P_is_symmetric", symmetric_error < 1e-12},
            {"trace_error", trace_error},
            {"trace_is_k", trace_error < 1e-12},
            {"residual_orthogonality", residual_orthogonality},
            {"residual_is_orthogonal", residual_orthogonality < 1e-12},
            {"pythagoras_error", pythagoras_error},
            {"pythagoras_holds", pythagoras_error < 1e-12},
            {"complement_is_projection", complement_error < 1e-12},
            {"input_v", std::vector<double>{v.x(), v.y(), v.z()}},
        };
        trace->record_matrix("projection", item_name + "/A", A,
                             "Spanning matrix A (columns span the subspace)", checks);
        trace->record_matrix("projection", item_name + "/P", res.P,
                             "Projector P = A (A^T A)^-1 A^T", checks);
        trace->record_vector("projection", item_name + "/parallel", res.parallel,
                             "P v: component of v inside the subspace", checks);
        trace->record_vector("projection", item_name + "/perpendicular", res.perpendicular,
                             "(I - P) v: component of v orthogonal to the subspace", checks);
    }
    return res;
}

NormalProjectionResult stage_projection_normal(const Eigen::Vector3d& v,
                                               const Eigen::Vector3d& normal, Trace* trace,
                                               const std::string& item_name) {
    const double v_norm = v.norm();
    if (v_norm == 0.0 || normal.norm() == 0.0) {
        throw std::invalid_argument("stage_projection_normal: zero-length input");
    }
    const Eigen::Vector3d n = normal.normalized();

    NormalProjectionResult res;
    res.P_normal = n * n.transpose();
    res.P_tangent = Eigen::Matrix3d::Identity() - res.P_normal;
    res.along_normal = res.P_normal * v;
    res.in_tangent_plane = res.P_tangent * v;
    res.reflected = v - 2.0 * res.along_normal;  // (I - 2 n n^T) v
    res.cosine = v.dot(n) / v_norm;

    if (trace != nullptr) {
        const double idempotent_error =
            (res.P_normal * res.P_normal - res.P_normal).cwiseAbs().maxCoeff();
        const double length_error = std::abs(res.reflected.norm() - v_norm);
        // Reflection keeps the tangent part and flips the normal part.
        const double flip_error = std::abs(res.reflected.dot(n) + v.dot(n));
        const double tangent_error =
            (res.reflected - (res.in_tangent_plane - res.along_normal)).cwiseAbs().maxCoeff();
        // Reflecting twice returns the original vector (the reflection is an involution).
        const Eigen::Vector3d twice = res.reflected - 2.0 * (res.P_normal * res.reflected);
        const double involution_error = (twice - v).cwiseAbs().maxCoeff();
        const double split_error =
            (res.along_normal + res.in_tangent_plane - v).cwiseAbs().maxCoeff();

        const std::map<std::string, nlohmann::json> checks = {
            {"idempotent_error", idempotent_error},
            {"P2_is_P", idempotent_error < 1e-12},
            {"length_error", length_error},
            {"reflection_preserves_length", length_error < 1e-12},
            {"flip_error", flip_error},
            {"normal_component_flipped", flip_error < 1e-12},
            {"tangent_error", tangent_error},
            {"tangent_component_kept", tangent_error < 1e-12},
            {"involution_error", involution_error},
            {"reflection_is_involution", involution_error < 1e-12},
            {"split_error", split_error},
            {"split_sums_to_v", split_error < 1e-12},
            {"cosine", res.cosine},
            {"input_v", std::vector<double>{v.x(), v.y(), v.z()}},
            {"input_normal", std::vector<double>{normal.x(), normal.y(), normal.z()}},
        };
        trace->record_matrix("projection", item_name + "/P_normal", res.P_normal,
                             "Projection onto the normal line, n n^T", checks);
        trace->record_matrix("projection", item_name + "/P_tangent", res.P_tangent,
                             "Projection onto the tangent plane, I - n n^T", checks);
        trace->record_vector("projection", item_name + "/along_normal", res.along_normal, "n n^T v",
                             checks);
        trace->record_vector("projection", item_name + "/in_tangent_plane", res.in_tangent_plane,
                             "(I - n n^T) v", checks);
        trace->record_vector("projection", item_name + "/reflected", res.reflected,
                             "Mirror direction v - 2 n n^T v", checks);
    }
    return res;
}

}  // namespace mfad
