#pragma once

#include "ray.hpp"

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Thin-lens Camera with depth of field support.
 * Follows right-handed convention with camera frame formed by orthogonal vectors (u, v, w).
 * When aperture > 0, rays originate from a disk around the lens center, producing
 * depth of field blur for objects not at the focus distance.
 */
class Camera {
public:
    Camera(const Eigen::Vector3f& look_from, const Eigen::Vector3f& look_at,
           const Eigen::Vector3f& up, float vfov_degrees, float aspect_ratio, float aperture = 0.0f,
           float focus_dist = -1.0f);

    /**
     * @brief Generates a ray for screen coordinates (s, t) in [0.0, 1.0].
     * (0, 0) is bottom-left, (1, 1) is top-right.
     * When aperture > 0, ray origins are jittered on the lens disk.
     */
    Ray generate_ray(float s, float t) const;

    const Eigen::Vector3f& origin() const { return origin_; }
    const Eigen::Vector3f& u() const { return u_; }
    const Eigen::Vector3f& v() const { return v_; }
    const Eigen::Vector3f& w() const { return w_; }
    float aperture() const { return aperture_; }
    float focus_dist() const { return focus_dist_; }

private:
    Eigen::Vector3f origin_;
    Eigen::Vector3f lower_left_corner_;
    Eigen::Vector3f horizontal_;
    Eigen::Vector3f vertical_;
    Eigen::Vector3f u_, v_, w_;
    float lens_radius_{0.0f};
    float aperture_{0.0f};
    float focus_dist_{1.0f};
};

}  // namespace mfad
