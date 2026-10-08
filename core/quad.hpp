#pragma once

#include "hittable.hpp"

#include <Eigen/Dense>
#include <memory>

namespace mfad {

/**
 * @brief Planar Quadrilateral primitive defined by corner point Q and edge vectors u, v.
 * Represents a parallelogram / rectangle in 3D space: P(alpha, beta) = Q + alpha * u + beta * v,
 * where alpha, beta in [0, 1].
 */
class Quad : public Hittable {
public:
    Quad(const Eigen::Vector3f& Q, const Eigen::Vector3f& u, const Eigen::Vector3f& v,
         const Eigen::Vector3f& color = Eigen::Vector3f::Ones(),
         std::shared_ptr<Material> material = nullptr);

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

    const Eigen::Vector3f& corner() const { return Q_; }
    const Eigen::Vector3f& u() const { return u_; }
    const Eigen::Vector3f& v() const { return v_; }
    const Eigen::Vector3f& normal() const { return normal_; }
    float area() const { return area_; }
    const Eigen::Vector3f& color() const { return color_; }
    std::shared_ptr<Material> material() const { return material_; }

private:
    Eigen::Vector3f Q_;
    Eigen::Vector3f u_;
    Eigen::Vector3f v_;
    Eigen::Vector3f normal_;
    Eigen::Vector3f w_;  // Projection vector for planar coordinate inversion
    float D_;            // Plane equation constant: normal . Q
    float area_;
    Eigen::Vector3f color_;
    std::shared_ptr<Material> material_;
};

}  // namespace mfad
