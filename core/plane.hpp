#pragma once

#include "hittable.hpp"

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Infinite 3D Plane representation with ray-plane intersection.
 * Defined by equation (P - point) . normal = 0.
 */
class Plane : public Hittable {
public:
    Plane(const Eigen::Vector3f& point, const Eigen::Vector3f& normal,
          const Eigen::Vector3f& color = Eigen::Vector3f::Ones(),
          std::shared_ptr<Material> material = nullptr);

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    AABB bounding_box() const override;

    const Eigen::Vector3f& point() const { return point_; }
    const Eigen::Vector3f& normal() const { return normal_; }
    const Eigen::Vector3f& color() const { return color_; }
    std::shared_ptr<Material> material() const { return material_; }

private:
    Eigen::Vector3f point_;
    Eigen::Vector3f normal_;
    Eigen::Vector3f color_;
    std::shared_ptr<Material> material_;
};

}  // namespace mfad
