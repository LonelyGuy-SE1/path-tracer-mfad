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

    // Local-to-World transform: v_world = Q * v_local
    inline Eigen::Vector3d to_world(const Eigen::Vector3d& local) const { return Q * local; }

    // World-to-Local transform: v_local = Q^T * v_world (Q^T = Q^-1 for orthogonal matrix)
    inline Eigen::Vector3d to_local(const Eigen::Vector3d& world) const {
        return Q.transpose() * world;
    }

    // Semantic accessors for camera frames
    inline const Eigen::Vector3d& right() const { return tangent; }
    inline const Eigen::Vector3d& up() const { return bitangent; }
    inline const Eigen::Vector3d& forward() const { return normal; }
};

/**
 * @brief Constructs an orthonormal basis (T, B, N) around a given surface normal.
 * Uses Duff et al. (2017) branchless, square-root free formulation.
 * Columns of Q satisfy: Q^T * Q = I, det(Q) = 1.
 *
 * @param normal Input surface normal vector (will be normalized).
 * @param trace Optional trace instance for logging (to avoid overhead per ray hit).
 * @param item_name Name of the record when logging to trace.
 */
BasisResult stage_basis_normal(const Eigen::Vector3d& normal, Trace* trace = nullptr,
                               const std::string& item_name = "normal_frame");

/**
 * @brief Constructs an orthonormal basis using the classical Gram-Schmidt process
 * given a normal vector and a guide/up direction vector.
 */
BasisResult stage_basis_gram_schmidt(const Eigen::Vector3d& normal, const Eigen::Vector3d& guide,
                                     Trace* trace = nullptr,
                                     const std::string& item_name = "gram_schmidt_frame");

/**
 * @brief Constructs an orthonormal camera basis (right, up, forward/view) from
 * look_from, look_at, and rough up vector using the Gram-Schmidt process (Issue #11).
 * Handles the singularity when up is parallel to the view direction.
 * Columns of Q satisfy: Q^T * Q = I, det(Q) = 1.
 *
 * @param look_from Eye position in world space.
 * @param look_at Target position in world space.
 * @param up Rough guide up vector (e.g., [0, 1, 0]).
 * @param trace Optional trace instance for logging.
 * @param item_name Name of the record when logging to trace.
 */
BasisResult stage_basis_camera(const Eigen::Vector3d& look_from, const Eigen::Vector3d& look_at,
                               const Eigen::Vector3d& up, Trace* trace = nullptr,
                               const std::string& item_name = "camera_frame");

}  // namespace mfad
