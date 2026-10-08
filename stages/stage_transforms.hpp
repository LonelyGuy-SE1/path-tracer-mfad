#pragma once

#include "stage_interface.hpp"
#include "trace.hpp"

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mfad {

/**
 * @brief Constructs a 4x4 translation matrix.
 */
Eigen::Matrix4d make_translate(const Eigen::Vector3d& t);

/**
 * @brief Constructs a 4x4 rotation matrix around the X axis.
 */
Eigen::Matrix4d make_rotate_x(double angle_rad);

/**
 * @brief Constructs a 4x4 rotation matrix around the Y axis.
 */
Eigen::Matrix4d make_rotate_y(double angle_rad);

/**
 * @brief Constructs a 4x4 rotation matrix around the Z axis.
 */
Eigen::Matrix4d make_rotate_z(double angle_rad);

/**
 * @brief Constructs a 4x4 rotation matrix around an arbitrary axis using Rodrigues' formula.
 */
Eigen::Matrix4d make_rotate_axis(const Eigen::Vector3d& axis, double angle_rad);

/**
 * @brief Constructs a 4x4 non-uniform scale matrix.
 */
Eigen::Matrix4d make_scale(const Eigen::Vector3d& s);

/**
 * @brief Constructs a 4x4 uniform scale matrix.
 */
Eigen::Matrix4d make_scale(double s);

/**
 * @brief Constructs a 4x4 3D shear matrix:
 * x' = x + s_xy * y + s_xz * z
 * y' = y + s_yx * x + s_yz * z
 * z' = z + s_zx * x + s_zy * y
 */
Eigen::Matrix4d make_shear(double s_xy = 0.0, double s_xz = 0.0, double s_yx = 0.0,
                           double s_yz = 0.0, double s_zx = 0.0, double s_zy = 0.0);

/**
 * @brief Composes a sequence of 4x4 affine matrices via sequential left multiplication.
 * Given [M1, M2, M3], the composite transformation is M = M3 * M2 * M1.
 */
Eigen::Matrix4d compose_transforms(const std::vector<Eigen::Matrix4d>& transforms);

/**
 * @brief Transforms a 3D point using homogeneous coordinates (w = 1, divides by w').
 */
Eigen::Vector3d transform_point(const Eigen::Matrix4d& M, const Eigen::Vector3d& p);

/**
 * @brief Transforms a 3D direction vector (w = 0).
 */
Eigen::Vector3d transform_vector(const Eigen::Matrix4d& M, const Eigen::Vector3d& v);

/**
 * @brief Transforms a surface normal vector using the inverse transpose M^{-T}.
 */
Eigen::Vector3d transform_normal(const Eigen::Matrix4d& M, const Eigen::Vector3d& n);

/**
 * @brief MFAD stage for 4x4 affine transformation analysis and linear algebra validation (Issue #10).
 * Validates orthogonal properties (Q^T Q = I and det(Q) = 1) for rotations, and logs
 * that scale and shear transformations fail these isometry checks.
 *
 * @param transform 4x4 transformation matrix.
 * @param transform_type Description ("rotation", "scale", "shear", "composite").
 * @param trace Optional trace instance for logging.
 * @param item_name Key name when written to trace records.
 */
StageResult<Eigen::Matrix4d> stage_transforms(const Eigen::Matrix4d& transform,
                                             const std::string& transform_type,
                                             Trace* trace = nullptr,
                                             const std::string& item_name = "transform");

}  // namespace mfad
