#include "stage_transforms.hpp"

#include <cmath>

namespace mfad {

Eigen::Matrix4d make_translate(const Eigen::Vector3d& t) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 3) = t.x();
    M(1, 3) = t.y();
    M(2, 3) = t.z();
    return M;
}

Eigen::Matrix4d make_rotate_x(double angle_rad) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    M(1, 1) = c;
    M(1, 2) = -s;
    M(2, 1) = s;
    M(2, 2) = c;
    return M;
}

Eigen::Matrix4d make_rotate_y(double angle_rad) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    M(0, 0) = c;
    M(0, 2) = s;
    M(2, 0) = -s;
    M(2, 2) = c;
    return M;
}

Eigen::Matrix4d make_rotate_z(double angle_rad) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    M(0, 0) = c;
    M(0, 1) = -s;
    M(1, 0) = s;
    M(1, 1) = c;
    return M;
}

Eigen::Matrix4d make_rotate_axis(const Eigen::Vector3d& axis, double angle_rad) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    Eigen::Vector3d a = axis.normalized();
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    double one_minus_c = 1.0 - c;

    M(0, 0) = c + a.x() * a.x() * one_minus_c;
    M(0, 1) = a.x() * a.y() * one_minus_c - a.z() * s;
    M(0, 2) = a.x() * a.z() * one_minus_c + a.y() * s;

    M(1, 0) = a.y() * a.x() * one_minus_c + a.z() * s;
    M(1, 1) = c + a.y() * a.y() * one_minus_c;
    M(1, 2) = a.y() * a.z() * one_minus_c - a.x() * s;

    M(2, 0) = a.z() * a.x() * one_minus_c - a.y() * s;
    M(2, 1) = a.z() * a.y() * one_minus_c + a.x() * s;
    M(2, 2) = c + a.z() * a.z() * one_minus_c;

    return M;
}

Eigen::Matrix4d make_scale(const Eigen::Vector3d& s) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 0) = s.x();
    M(1, 1) = s.y();
    M(2, 2) = s.z();
    return M;
}

Eigen::Matrix4d make_scale(double s) {
    return make_scale(Eigen::Vector3d(s, s, s));
}

Eigen::Matrix4d make_shear(double s_xy, double s_xz, double s_yx, double s_yz, double s_zx,
                           double s_zy) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 1) = s_xy;
    M(0, 2) = s_xz;
    M(1, 0) = s_yx;
    M(1, 2) = s_yz;
    M(2, 0) = s_zx;
    M(2, 1) = s_zy;
    return M;
}

Eigen::Matrix4d compose_transforms(const std::vector<Eigen::Matrix4d>& transforms) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    for (const auto& T : transforms) {
        M = T * M;
    }
    return M;
}

Eigen::Vector3d transform_point(const Eigen::Matrix4d& M, const Eigen::Vector3d& p) {
    Eigen::Vector4d p4(p.x(), p.y(), p.z(), 1.0);
    Eigen::Vector4d res = M * p4;
    if (std::abs(res.w()) > 1e-12) {
        return res.head<3>() / res.w();
    }
    return res.head<3>();
}

Eigen::Vector3d transform_vector(const Eigen::Matrix4d& M, const Eigen::Vector3d& v) {
    return M.block<3, 3>(0, 0) * v;
}

Eigen::Vector3d transform_normal(const Eigen::Matrix4d& M, const Eigen::Vector3d& n) {
    Eigen::Matrix3d normal_matrix = M.block<3, 3>(0, 0).inverse().transpose();
    return (normal_matrix * n).normalized();
}

StageResult<Eigen::Matrix4d> stage_transforms(const Eigen::Matrix4d& transform,
                                             const std::string& transform_type,
                                             Trace* trace, const std::string& item_name) {
    // Extract upper-left 3x3 block Q for linear algebra invariant analysis
    Eigen::Matrix3d Q = transform.block<3, 3>(0, 0);

    Eigen::Matrix3d QtQ = Q.transpose() * Q;
    double QtQ_error = (QtQ - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    bool QtQ_is_identity = (QtQ_error < 1e-12);

    double det = Q.determinant();
    bool det_is_one = (std::abs(det - 1.0) < 1e-12);

    bool is_rotation = (transform_type == "rotation" || transform_type == "rotate_x" ||
                        transform_type == "rotate_y" || transform_type == "rotate_z" ||
                        transform_type == "rotate_axis");

    bool passes_isometry = (QtQ_is_identity && det_is_one);

    if (trace != nullptr) {
        trace->record_matrix("transforms", item_name, transform,
                             "4x4 transformation matrix analysis: " + transform_type,
                             {{"transform_type", transform_type},
                              {"QtQ_error", QtQ_error},
                              {"QtQ_is_identity", QtQ_is_identity},
                              {"det", det},
                              {"det_is_one", det_is_one},
                              {"is_rotation", is_rotation},
                              {"passes_isometry", passes_isometry}});
    }

    return StageResult<Eigen::Matrix4d>(transform, true, "transforms",
                                       "Transform stage analysis complete (" + transform_type + ")");
}

}  // namespace mfad
