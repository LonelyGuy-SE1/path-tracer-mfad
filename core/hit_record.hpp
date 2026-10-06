#pragma once

#include "ray.hpp"

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Stores ray-surface intersection information.
 */
struct HitRecord {
    float t{0.0f};
    Eigen::Vector3f point{Eigen::Vector3f::Zero()};
    Eigen::Vector3f normal{0.0f, 1.0f, 0.0f};
    bool front_face{true};
    Eigen::Vector3f color{1.0f, 1.0f, 1.0f};

    inline void set_face_normal(const Ray& r, const Eigen::Vector3f& outward_normal) {
        front_face = r.direction.dot(outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

}  // namespace mfad
