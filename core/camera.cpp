#include "camera.hpp"

#include "stage_basis.hpp"

#include <cmath>

namespace mfad {

Camera::Camera(const Eigen::Vector3f& look_from, const Eigen::Vector3f& look_at,
               const Eigen::Vector3f& up, float vfov_degrees, float aspect_ratio) {
    origin_ = look_from;

    const float pi = 3.14159265358979323846f;
    float theta = vfov_degrees * pi / 180.0f;
    float half_height = std::tan(theta / 2.0f);
    float half_width = aspect_ratio * half_height;

    // Use stage_basis_camera (#11) to construct orthonormal camera frame Q in double precision
    BasisResult basis =
        stage_basis_camera(look_from.cast<double>(), look_at.cast<double>(), up.cast<double>());
    u_ = basis.right().cast<float>();
    v_ = basis.up().cast<float>();
    w_ = basis.forward().cast<float>();

    horizontal_ = 2.0f * half_width * u_;
    vertical_ = 2.0f * half_height * v_;
    lower_left_corner_ = origin_ - half_width * u_ - half_height * v_ - w_;
}

Ray Camera::generate_ray(float s, float t) const {
    Eigen::Vector3f target = lower_left_corner_ + s * horizontal_ + t * vertical_;
    return Ray(origin_, target - origin_);
}

}  // namespace mfad
