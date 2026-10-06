#include "sphere.hpp"

#include <algorithm>
#include <cmath>

namespace mfad {

Sphere::Sphere(const Eigen::Vector3f& center, float radius, const Eigen::Vector3f& color,
               std::shared_ptr<Material> material)
    : center_(center), radius_(radius), color_(color), material_(std::move(material)) {}

bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Eigen::Vector3f oc = r.origin - center_;
    float a = r.direction.squaredNorm();
    float half_b = oc.dot(r.direction);
    float c = oc.squaredNorm() - radius_ * radius_;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    float sqrt_d = std::sqrt(discriminant);
    float root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.point = r.point_at(rec.t);
    Eigen::Vector3f outward_normal = (rec.point - center_) / radius_;
    rec.set_face_normal(r, outward_normal.normalized());
    rec.color = color_;
    rec.material = material_.get();
    return true;
}

}  // namespace mfad
