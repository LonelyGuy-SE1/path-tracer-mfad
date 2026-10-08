#pragma once

#include <Eigen/Dense>
#include <cassert>

namespace mfad {

/**
 * @brief Represents a 3D ray r(t) = origin + t * direction.
 * Uses float precision for performance inside the render loop.
 */
struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f direction;

    Ray() : origin(0.0f, 0.0f, 0.0f), direction(0.0f, 0.0f, 1.0f) {}
    Ray(const Eigen::Vector3f& orig, const Eigen::Vector3f& dir) : origin(orig) {
        assert(dir.squaredNorm() > 1e-12f && "Ray direction cannot be zero");
        direction = dir.normalized();
    }

    inline Eigen::Vector3f point_at(float t) const { return origin + t * direction; }
    inline Eigen::Vector3f at(float t) const { return point_at(t); }
};

}  // namespace mfad
