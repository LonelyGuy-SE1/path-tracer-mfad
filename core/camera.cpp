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

    // Orthonormal camera frame with guard for up parallel to look direction
    w_ = (look_from - look_at).normalized();
    Eigen::Vector3f u_cross = up.cross(w_);
    if (u_cross.squaredNorm() < 1e-6f) {
        // Fallback reference axis when looking straight along up direction
        Eigen::Vector3f alt_up = (std::abs(w_.z()) < 0.9f) ? Eigen::Vector3f(0.0f, 0.0f, 1.0f)
                                                           : Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        u_cross = alt_up.cross(w_);
    }
    u_ = u_cross.normalized();
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
