#pragma once

#include "hittable.hpp"
#include "material.hpp"

#include <Eigen/Dense>
#include <memory>

namespace mfad {

/**
 * @brief Ellipsoid primitive with arbitrary orientation and quadric matrix representation.
 */
class Ellipsoid : public Hittable {
public:
    Ellipsoid(const Eigen::Vector3f& center, const Eigen::Vector3f& radii,
              const Eigen::Matrix3f& rotation = Eigen::Matrix3f::Identity(),
              const Eigen::Vector3f& color = Eigen::Vector3f::Ones(),
              std::shared_ptr<Material> material = nullptr);

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    AABB bounding_box() const override;

    const Eigen::Vector3f& center() const { return center_; }
    const Eigen::Vector3f& radii() const { return radii_; }
    const Eigen::Matrix3f& rotation() const { return rotation_; }
    const Eigen::Matrix4d& quadric_matrix() const { return quadric_matrix_; }
    const std::shared_ptr<Material>& material() const { return material_; }

private:
    Eigen::Vector3f center_;
    Eigen::Vector3f radii_;
    Eigen::Matrix3f rotation_;
    Eigen::Matrix3f inv_rotation_scale_;  // M^-1 = diag(1/r) * R^T
    Eigen::Matrix3f a_mat_;               // A = R * diag(1/r^2) * R^T
    Eigen::Matrix4d quadric_matrix_;      // 4x4 algebraic quadric matrix
    Eigen::Vector3f color_;
    std::shared_ptr<Material> material_;
    const Material* raw_material_{nullptr};
};

}  // namespace mfad
