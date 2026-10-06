#include "plane.hpp"

#include <algorithm>
#include <cmath>

namespace mfad {

Plane::Plane(const Eigen::Vector3f& point, const Eigen::Vector3f& normal,
             const Eigen::Vector3f& color, std::shared_ptr<Material> material)
    : point_(point), normal_(normal.normalized()), color_(color), material_(std::move(material)) {}

bool Plane::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    float denom = r.direction.dot(normal_);

    // Check if ray is roughly parallel to the plane
    if (std::abs(denom) < 1e-6f) {
        return false;
    }

    float t = (point_ - r.origin).dot(normal_) / denom;
    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.point = r.point_at(t);
    rec.set_face_normal(r, normal_);
    rec.color = color_;
    rec.material = material_.get();
    return true;
}

}  // namespace mfad
