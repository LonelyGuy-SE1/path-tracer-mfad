#include "camera.hpp"

#include "stage_basis.hpp"

#include <cmath>
#include <random>

namespace mfad {

namespace {
inline Eigen::Vector2f random_in_unit_disk() {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    while (true) {
        Eigen::Vector2f p(2.0f * dis(gen) - 1.0f, 2.0f * dis(gen) - 1.0f);
        if (p.squaredNorm() < 1.0f)
            return p;
    }
}
}  // namespace

Camera::Camera(const Eigen::Vector3f& look_from, const Eigen::Vector3f& look_at,
               const Eigen::Vector3f& up, float vfov_degrees, float aspect_ratio, float aperture,
               float focus_dist)
    : aperture_(aperture) {
    origin_ = look_from;
    lens_radius_ = aperture / 2.0f;

    // Auto-compute focus distance if not specified
    if (focus_dist < 0.0f) {
        focus_dist_ = (look_from - look_at).norm();
    } else {
        focus_dist_ = focus_dist;
    }

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

    horizontal_ = 2.0f * half_width * focus_dist_ * u_;
    vertical_ = 2.0f * half_height * focus_dist_ * v_;
    lower_left_corner_ =
        origin_ - half_width * focus_dist_ * u_ - half_height * focus_dist_ * v_ - focus_dist_ * w_;
}

Ray Camera::generate_ray(float s, float t) const {
    Eigen::Vector3f target = lower_left_corner_ + s * horizontal_ + t * vertical_;
    if (lens_radius_ > 0.0f) {
        Eigen::Vector2f rd = lens_radius_ * random_in_unit_disk();
        Eigen::Vector3f offset = u_ * rd.x() + v_ * rd.y();
        return Ray(origin_ + offset, target - origin_ - offset);
    }
    return Ray(origin_, target - origin_);
}

}  // namespace mfad
