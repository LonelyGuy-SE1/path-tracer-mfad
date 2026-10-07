#include "stage_camera.hpp"

#include "stage_basis.hpp"
#include "stage_projection.hpp"
#include "trace.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace mfad {

Camera::Camera(const Eigen::Vector3d& look_from_,
               const Eigen::Vector3d& look_at_,
               const Eigen::Vector3d& world_up_,
               double vertical_fov_degrees_,
               double aspect_ratio_)
    : look_from(look_from_),
      look_at(look_at_),
      world_up(world_up_),
      vertical_fov_degrees(vertical_fov_degrees_),
      aspect_ratio(aspect_ratio_) {
    if ((look_at - look_from).norm() == 0.0) {
        throw std::invalid_argument("Camera: look_from and look_at must differ");
    }
    if (world_up.norm() == 0.0) {
        throw std::invalid_argument("Camera: world_up must be non-zero");
    }
    if (vertical_fov_degrees <= 0.0 || vertical_fov_degrees >= 180.0) {
        throw std::invalid_argument("Camera: vertical FOV must be in (0, 180)");
    }
    if (aspect_ratio <= 0.0) {
        throw std::invalid_argument("Camera: aspect ratio must be positive");
    }
}

CameraRayResult Camera::generate_ray(double u, double v,
                                     Trace* trace,
                                     const std::string& item_name) const {
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        throw std::invalid_argument("Camera::generate_ray: u and v must be in [0, 1]");
    }

    // Forward direction is the camera normal N.
    const Eigen::Vector3d forward = (look_at - look_from).normalized();

    // Basis Q = [right, camera_up, forward].
    const BasisResult basis =
        stage_basis_gram_schmidt(forward, world_up, trace, item_name + "/basis");

    const Eigen::Vector3d right = basis.Q.col(0);
    const Eigen::Vector3d camera_up = basis.Q.col(1);

    // Convert [0,1] image coordinates to centered image-plane coordinates.
    // u=0/v=0 is the lower-left corner; u=0.5/v=0.5 is the image center.
    const double theta = vertical_fov_degrees * std::numbers::pi / 180.0;
    const double half_height = std::tan(theta * 0.5);
    const double half_width = aspect_ratio * half_height;

    const double x = (2.0 * u - 1.0) * half_width;
    const double y = (2.0 * v - 1.0) * half_height;

    // A=[right,up] spans the camera image plane.
    Eigen::MatrixXd A(3, 2);
    A.col(0) = right;
    A.col(1) = camera_up;

    const Eigen::Vector3d image_offset = x * right + y * camera_up;
    const ProjectionResult projection =
        stage_projection_subspace(A, image_offset, trace, item_name + "/image_plane");

    // The projected vector is the image-plane part. Add the forward direction
    // to form the ray and normalize it.
    const Eigen::Vector3d ray_direction =
        (forward + projection.parallel).normalized();

    CameraRayResult result;
    result.origin = look_from;
    result.direction = ray_direction;
    result.image_plane_component = projection.parallel;
    result.image_plane_projector = projection.P;

    return result;
}

}  // namespace mfad
