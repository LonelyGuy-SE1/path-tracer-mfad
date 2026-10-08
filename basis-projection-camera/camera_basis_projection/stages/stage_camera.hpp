#pragma once

#include <Eigen/Dense>
#include <string>

namespace mfad {

class Trace;

/**
 * Small camera construction built from the Basis + Projection stages.
 *
 * The camera frame is:
 *   right = Q.col(0)
 *   up    = Q.col(1)
 *   forward = Q.col(2)
 *
 * Q is produced by stage_basis_gram_schmidt(forward, world_up).
 *
 * The image-plane vector is projected onto A = [right, up] using
 * stage_projection_subspace().
 */
struct CameraRayResult {
    Eigen::Vector3d origin;
    Eigen::Vector3d direction;
    Eigen::Vector3d image_plane_component;
    Eigen::Matrix3d image_plane_projector;
};

struct Camera {
    Eigen::Vector3d look_from;
    Eigen::Vector3d look_at;
    Eigen::Vector3d world_up;
    double vertical_fov_degrees;
    double aspect_ratio;

    Camera(const Eigen::Vector3d& look_from,
           const Eigen::Vector3d& look_at,
           const Eigen::Vector3d& world_up,
           double vertical_fov_degrees,
           double aspect_ratio);

    CameraRayResult generate_ray(double u, double v,
                                 Trace* trace = nullptr,
                                 const std::string& item_name = "camera") const;
};

}  // namespace mfad
