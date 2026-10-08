#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <string>

namespace mfad {

class Trace;

struct BasisResult {
    Eigen::Vector3d tangent;
    Eigen::Vector3d bitangent;
    Eigen::Vector3d normal;
    Eigen::Matrix3d Q;  // Orthogonal matrix: columns are [tangent, bitangent, normal]

    // Local-to-world: v_world = Q * v_local
    inline Eigen::Vector3d to_world(const Eigen::Vector3d& v_local) const {
        return Q * v_local;
    }

    // World-to-local: v_local = Q^T * v_world (Q^T = Q^-1 for an orthogonal matrix)
    inline Eigen::Vector3d to_local(const Eigen::Vector3d& v_world) const {
        return Q.transpose() * v_world;
    }

    // Semantic accessors for camera frames
    inline const Eigen::Vector3d& right() const { return tangent; }
    inline const Eigen::Vector3d& up() const { return bitangent; }
    inline const Eigen::Vector3d& forward() const { return normal; }
};

/**
 * Constructs an orthonormal basis (T, B, N) around a surface normal.
 * Columns of Q satisfy Q^T * Q = I, det(Q) = 1.
 */
BasisResult stage_basis_normal(const Eigen::Vector3d& normal,
                               Trace* trace = nullptr,
                               const std::string& item_name = "basis");

/**
 * Constructs an orthonormal basis with Gram-Schmidt, given a normal
 * and a guide/up direction.
 */
BasisResult stage_basis_gram_schmidt(const Eigen::Vector3d& normal,
                                     const Eigen::Vector3d& guide,
                                     Trace* trace = nullptr,
                                     const std::string& item_name = "gram_schmidt_frame");

/**
 * Constructs an orthonormal camera basis (right, up, forward) from
 * look_from, look_at and a rough up vector.
 */
BasisResult stage_basis_camera(const Eigen::Vector3d& look_from,
                               const Eigen::Vector3d& look_at,
                               const Eigen::Vector3d& up,
                               Trace* trace = nullptr,
                               const std::string& item_name = "camera_frame");

}  // namespace mfad