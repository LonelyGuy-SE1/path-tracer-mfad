#pragma once

#include "hittable.hpp"
#include "material.hpp"

#include <Eigen/Dense>
#include <memory>

namespace mfad {

/**
 * @brief Triangle primitive with analytical Möller-Trumbore ray intersection.
 */
class Triangle : public Hittable {
public:
    Triangle(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
             const Eigen::Vector3f& color = Eigen::Vector3f::Ones(),
             std::shared_ptr<Material> material = nullptr);

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

    const Eigen::Vector3f& v0() const { return v0_; }
    const Eigen::Vector3f& v1() const { return v1_; }
    const Eigen::Vector3f& v2() const { return v2_; }
    const Eigen::Vector3f& normal() const { return normal_; }
    const std::shared_ptr<Material>& material() const { return material_; }

private:
    Eigen::Vector3f v0_;
    Eigen::Vector3f v1_;
    Eigen::Vector3f v2_;
    Eigen::Vector3f e1_;
    Eigen::Vector3f e2_;
    Eigen::Vector3f normal_;
    Eigen::Vector3f color_;
    std::shared_ptr<Material> material_;
    const Material* raw_material_{nullptr};
};

}  // namespace mfad
