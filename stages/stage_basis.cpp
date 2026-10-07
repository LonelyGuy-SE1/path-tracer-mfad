#include "stage_basis.hpp"
#include "trace.hpp"

#include <cmath>

namespace mfad {

BasisResult stage_basis_normal(const Eigen::Vector3d& normal, Trace* trace,
                               const std::string& item_name) {
    Eigen::Vector3d n = normal.normalized();

    const double sign = std::copysign(1.0, n.z());
    const double a = -1.0 / (sign + n.z());
    const double f = n.x() * n.y() * a;

    Eigen::Vector3d t(1.0 + sign * n.x() * n.x() * a, sign * f, -sign * n.x());
    Eigen::Vector3d b(f, sign + n.y() * n.y() * a, -n.y());

    BasisResult res;
    res.tangent = t;
    res.bitangent = b;
    res.normal = n;

    res.Q.col(0) = t;
    res.Q.col(1) = b;
    res.Q.col(2) = n;

    if (trace != nullptr) {
        Eigen::Matrix3d QtQ = res.Q.transpose() * res.Q;
        double QtQ_error =
            (QtQ - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
        bool QtQ_is_identity = (QtQ_error < 1e-12);

        double det = res.Q.determinant();
        bool det_is_one = (std::abs(det - 1.0) < 1e-12);

        Eigen::Vector3d v_test(1.0, 2.0, -3.0);
        v_test.normalize();
        Eigen::Vector3d v_world = res.to_world(v_test);
        Eigen::Vector3d v_recovered = res.to_local(v_world);
        double roundtrip_error =
            (v_recovered - v_test).cwiseAbs().maxCoeff();
        bool roundtrip_is_identity = (roundtrip_error < 1e-12);

        trace->record_matrix(
            "basis", item_name, res.Q,
            "Gram-Schmidt orthonormal basis [T, B, N]",
            {{"QtQ_error", QtQ_error},
             {"QtQ_is_identity", QtQ_is_identity},
             {"det", det},
             {"det_is_one", det_is_one},
             {"roundtrip_error", roundtrip_error},
             {"roundtrip_is_identity", roundtrip_is_identity}});
    }

    return res;
}

BasisResult stage_basis_camera(const Eigen::Vector3d& look_from,
                               const Eigen::Vector3d& look_at,
                               const Eigen::Vector3d& up,
                               Trace* trace,
                               const std::string& item_name) {
    Eigen::Vector3d view_dir = look_from - look_at;

    if (view_dir.squaredNorm() < 1e-12) {
        view_dir = Eigen::Vector3d(0.0, 0.0, 1.0);
    }

    Eigen::Vector3d w = view_dir.normalized();

    Eigen::Vector3d u_cross = up.cross(w);
    bool parallel_singularity = (u_cross.squaredNorm() < 1e-12);

    if (parallel_singularity) {
        Eigen::Vector3d alt_up =
            (std::abs(w.z()) < 0.9)
                ? Eigen::Vector3d(0.0, 0.0, 1.0)
                : Eigen::Vector3d(1.0, 0.0, 0.0);
        u_cross = alt_up.cross(w);
    }

    Eigen::Vector3d u = u_cross.normalized();
    Eigen::Vector3d v = w.cross(u);

    BasisResult res;
    res.tangent = u;
    res.bitangent = v;
    res.normal = w;

    res.Q.col(0) = u;
    res.Q.col(1) = v;
    res.Q.col(2) = w;

    if (trace != nullptr) {
        Eigen::Matrix3d QtQ = res.Q.transpose() * res.Q;
        double QtQ_error =
            (QtQ - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
        bool QtQ_is_identity = (QtQ_error < 1e-12);

        double det = res.Q.determinant();
        bool det_is_one = (std::abs(det - 1.0) < 1e-12);

        Eigen::Vector3d v_test(1.0, 0.5, -2.0);
        v_test.normalize();
        Eigen::Vector3d v_world = res.to_world(v_test);
        Eigen::Vector3d v_recovered = res.to_local(v_world);
        double roundtrip_error =
            (v_recovered - v_test).cwiseAbs().maxCoeff();
        bool roundtrip_is_identity = (roundtrip_error < 1e-12);

        trace->record_matrix(
            "basis", item_name, res.Q,
            "Gram-Schmidt camera orthonormal basis [right, up, forward]",
            {{"QtQ_error", QtQ_error},
             {"QtQ_is_identity", QtQ_is_identity},
             {"det", det},
             {"det_is_one", det_is_one},
             {"roundtrip_error", roundtrip_error},
             {"roundtrip_is_identity", roundtrip_is_identity},
             {"parallel_singularity_handled", parallel_singularity}});
    }

    return res;
}

}  // namespace mfad
