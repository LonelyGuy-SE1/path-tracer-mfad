#pragma once

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Point light source emitting isotropic radiant intensity.
 */
struct PointLight {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f intensity{1.0f, 1.0f, 1.0f};
    float radius{0.0f};

    PointLight() = default;
    PointLight(const Eigen::Vector3f& pos, const Eigen::Vector3f& inten, float rad = 0.0f)
        : position(pos), intensity(inten), radius(rad) {}
};

/**
 * @brief Directional light source (distant light at infinity).
 */
struct DirectionalLight {
    Eigen::Vector3f direction{0.0f, -1.0f, 0.0f};  // Vector pointing in direction of propagation
    Eigen::Vector3f intensity{1.0f, 1.0f, 1.0f};

    DirectionalLight() = default;
    DirectionalLight(const Eigen::Vector3f& dir, const Eigen::Vector3f& inten)
        : direction(dir.normalized()), intensity(inten) {}
};

}  // namespace mfad
