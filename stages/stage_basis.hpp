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

}  // namespace mfad
