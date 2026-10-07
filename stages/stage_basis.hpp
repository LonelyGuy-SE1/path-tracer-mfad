#pragma once

#include <Eigen/Dense>
#include <string>

namespace mfad {

class Trace;

struct BasisResult {
    Eigen::Vector3d tangent;
    Eigen::Vector3d bitangent;
    Eigen::Vector3d normal;
    Eigen::Matrix3d Q;

    Eigen::Vector3d to_world(const Eigen::Vector3d& v_local) const {
        return Q * v_local;
    }

    Eigen::Vector3d to_local(const Eigen::Vector3d& v_world) const {
        return Q.transpose() * v_world;
    }
};

BasisResult stage_basis_normal(const Eigen::Vector3d& normal,
                               Trace* trace = nullptr,
                               const std::string& item_name = "basis");

BasisResult stage_basis_camera(
    const Eigen::Vector3d& look_from,
    const Eigen::Vector3d& look_at,
    const Eigen::Vector3d& up,
    Trace* trace = nullptr,
    const std::string& item_name = "camera_frame");

}  // namespace mfad
