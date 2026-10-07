#include "stage_camera.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    mfad::Camera camera(
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, -1.0),
        Eigen::Vector3d(0.0, 1.0, 0.0),
        60.0,
        16.0 / 9.0);

    const auto center = camera.generate_ray(0.5, 0.5);
    const auto left = camera.generate_ray(0.0, 0.5);
    const auto right = camera.generate_ray(1.0, 0.5);
    const auto top = camera.generate_ray(0.5, 1.0);

    const double center_error =
        (center.direction - Eigen::Vector3d(0.0, 0.0, -1.0)).norm();

    if (center_error > 1e-12) {
        throw std::runtime_error("center ray is not aligned with camera forward");
    }

    if (std::abs(left.direction.norm() - 1.0) > 1e-12 ||
        std::abs(right.direction.norm() - 1.0) > 1e-12 ||
        std::abs(top.direction.norm() - 1.0) > 1e-12) {
        throw std::runtime_error("camera rays are not normalized");
    }

    if (left.direction.x() >= center.direction.x() ||
        right.direction.x() <= center.direction.x() ||
        top.direction.y() <= center.direction.y()) {
        throw std::runtime_error("camera image-plane directions are incorrect");
    }

    std::cout << "[PASS] Basis + Projection camera tests passed.\n";
    std::cout << "center = " << center.direction.transpose() << '\n';
    std::cout << "left   = " << left.direction.transpose() << '\n';
    std::cout << "right  = " << right.direction.transpose() << '\n';
    std::cout << "top    = " << top.direction.transpose() << '\n';
    return 0;
}
