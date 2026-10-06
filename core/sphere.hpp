#pragma once

#include "hittable.hpp"

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Analytical Sphere representation with ray-sphere intersection.
 */
class Sphere : public Hittable {
public:
    Sphere(const Eigen::Vector3f& center, float radius,
           const Eigen::Vector3f& color = Eigen::Vector3f::Ones());

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

    const Eigen::Vector3f& center() const { return center_; }
    float radius() const { return radius_; }
    const Eigen::Vector3f& color() const { return color_; }

private:
    Eigen::Vector3f center_;
    float radius_;
    Eigen::Vector3f color_;
};

}  // namespace mfad
