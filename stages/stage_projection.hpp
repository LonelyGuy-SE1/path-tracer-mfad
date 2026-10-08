#pragma once

#include "stage_interface.hpp"
#include "trace.hpp"

#include <Eigen/Dense>
#include <string>

namespace mfad {

struct ShadowRayResult {
    Eigen::Vector3d direction;
    double distance;
    Eigen::Vector3d shadow_origin;
};

struct ProjectResult {
    Eigen::Matrix4d projection_matrix;
    Eigen::Vector3d ndc;
    Eigen::Vector2d pixel;
    bool in_frustum;
};

/**
 * @brief MFAD stage computing orthogonal vector reflection (Issue #13).
 * Formula: r = v - 2 * (v . n) * n
 */
StageResult<Eigen::Vector3d> stage_reflect(const Eigen::Vector3d& v, const Eigen::Vector3d& n,
                                         Trace* trace = nullptr,
                                         const std::string& item_name = "reflect");

/**
 * @brief MFAD stage computing Lambertian diffuse cosine term (Issue #13).
 * Formula: max(0.0, normal . light_dir)
 */
StageResult<double> stage_diffuse_term(const Eigen::Vector3d& normal,
                                       const Eigen::Vector3d& light_dir, Trace* trace = nullptr,
                                       const std::string& item_name = "diffuse");

/**
 * @brief MFAD stage computing shadow ray test vector and surface offset (Issue #13).
 */
StageResult<ShadowRayResult> stage_shadow_direction(const Eigen::Vector3d& surface_point,
                                                   const Eigen::Vector3d& light_pos,
                                                   double eps = 1e-4, Trace* trace = nullptr,
                                                   const std::string& item_name = "shadow");

/**
 * @brief MFAD stage computing pinhole perspective projection onto the image plane (Issue #13).
 * Maps view space points through standard 4x4 perspective projection matrix into NDC and pixel coordinates.
 */
StageResult<ProjectResult> stage_project_image_plane(const Eigen::Vector3d& point_view_space,
                                                     double vfov_deg, double aspect_ratio,
                                                     int width, int height, double near_plane = 0.1,
                                                     double far_plane = 1000.0,
                                                     Trace* trace = nullptr,
                                                     const std::string& item_name = "project");

}  // namespace mfad
