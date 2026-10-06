#include "camera.hpp"

#include <cmath>

namespace mfad {

Camera::Camera(const Eigen::Vector3f& look_from, const Eigen::Vector3f& look_at,
               const Eigen::Vector3f& up, float vfov_degrees, float aspect_ratio) {
    origin_ = look_from;

    const float pi = 3.14159265358979323846f;
    float theta = vfov_degrees * pi / 180.0f;
    float half_height = std::tan(theta / 2.0f);
    float half_width = aspect_ratio * half_height;

    // Orthonormal camera frame
    w_ = (look_from - look_at).normalized();
    u_ = up.cross(w_).normalized();
    v_ = w_.cross(u_);

    horizontal_ = 2.0f * half_width * u_;
    vertical_ = 2.0f * half_height * v_;
    lower_left_corner_ = origin_ - half_width * u_ - half_height * v_ - w_;
}

Ray Camera::generate_ray(float s, float t) const {
    Eigen::Vector3f target = lower_left_corner_ + s * horizontal_ + t * vertical_;
    return Ray(origin_, target - origin_);
}

}  // namespace mfad
