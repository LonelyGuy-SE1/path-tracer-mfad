#pragma once

#include <Eigen/Dense>
#include <string>

namespace mfad {

class Trace;

/**
 * Orthogonal projection of a vector v onto the column space of A (3 x k, full column rank).
 *
 *   P = A (A^T A)^-1 A^T      v_par = P v      v_perp = (I - P) v
 *
 * P is symmetric and idempotent (P^2 = P), trace(P) = k, and A^T v_perp = 0.
 */
struct ProjectionResult {
    Eigen::Matrix3d P;
    Eigen::Vector3d parallel;
    Eigen::Vector3d perpendicular;
};

/**
 * @brief Projects v onto span(A). Columns of A must be linearly independent.
 * Used for the image plane (A = [right, up]) and as the general case of the two helpers below.
 *
 * @param A 3 x k matrix whose columns span the subspace (k = 1, 2 or 3).
 * @param v Vector to project.
 * @param trace Optional trace for logging.
 * @param item_name Prefix of the record names when logging.
 */
ProjectionResult stage_projection_subspace(const Eigen::MatrixXd& A, const Eigen::Vector3d& v,
                                           Trace* trace = nullptr,
                                           const std::string& item_name = "subspace_projection");

struct NormalProjectionResult {
    Eigen::Matrix3d P_normal;   // n n^T: projection onto the normal line
    Eigen::Matrix3d P_tangent;  // I - n n^T: projection onto the tangent plane
    Eigen::Vector3d along_normal;
    Eigen::Vector3d in_tangent_plane;
    Eigen::Vector3d reflected;  // v - 2 P_normal v
    double cosine;              // (v . n) / |v|, signed length of the projection on n
};

/**
 * @brief Splits v against a surface normal. This is the projection the renderer uses at a hit:
 * reflection (r = v - 2 (n n^T) v), the cosine for diffuse shading, and the tangent component.
 *
 * @param v Incoming direction (any length, nonzero).
 * @param normal Surface normal (will be normalized).
 */
NormalProjectionResult stage_projection_normal(const Eigen::Vector3d& v,
                                               const Eigen::Vector3d& normal,
                                               Trace* trace = nullptr,
                                               const std::string& item_name = "normal_projection");

}  // namespace mfad
