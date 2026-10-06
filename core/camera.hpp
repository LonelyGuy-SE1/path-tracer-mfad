#pragma once

#include "ray.hpp"

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Pinhole Camera generating rays through normalized viewport coordinates.
 * Follows right-handed convention with camera frame formed by orthogonal vectors (u, v, w).
 */
class Camera {
public:
    Camera(const Eigen::Vector3f& look_from, const Eigen::Vector3f& look_at,
           const Eigen::Vector3f& up, float vfov_degrees, float aspect_ratio);

    /**
     * @brief Generates a ray for screen coordinates (s, t) in [0.0, 1.0].
     * (0, 0) is bottom-left, (1, 1) is top-right.
     */
    Ray generate_ray(float s, float t) const;

    const Eigen::Vector3f& origin() const { return origin_; }
    const Eigen::Vector3f& u() const { return u_; }
    const Eigen::Vector3f& v() const { return v_; }
    const Eigen::Vector3f& w() const { return w_; }

private:
    Eigen::Vector3f origin_;
    Eigen::Vector3f lower_left_corner_;
    Eigen::Vector3f horizontal_;
    Eigen::Vector3f vertical_;
    Eigen::Vector3f u_, v_, w_;
};

}  // namespace mfad
