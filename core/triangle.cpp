#include "triangle.hpp"

#include <cmath>

namespace mfad {

Triangle::Triangle(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2,
                   const Eigen::Vector3f& color, std::shared_ptr<Material> material)
    : v0_(v0), v1_(v1), v2_(v2), e1_(v1 - v0), e2_(v2 - v0), color_(color),
      material_(std::move(material)), raw_material_(material_ ? material_.get() : nullptr) {
    Eigen::Vector3f n = e1_.cross(e2_);
    float len = n.norm();
    normal_ = (len > 1e-8f) ? (n / len) : Eigen::Vector3f(0.0f, 1.0f, 0.0f);
}

bool Triangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Eigen::Vector3f pvec = r.direction.cross(e2_);
    float det = e1_.dot(pvec);

    if (std::abs(det) < 1e-8f) {
        return false;
    }

    float inv_det = 1.0f / det;
    Eigen::Vector3f tvec = r.origin - v0_;
    float u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    Eigen::Vector3f qvec = tvec.cross(e1_);
    float v = r.direction.dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = e2_.dot(qvec) * inv_det;
    if (t < t_min || t > t_max) {
        return false;
    }

    rec.t = t;
    rec.point = r.at(t);
    rec.color = color_;
    rec.material = raw_material_;
    rec.set_face_normal(r, normal_);
    return true;
}

}  // namespace mfad

namespace mfad {
AABB Triangle::bounding_box() const {
    Eigen::Vector3f min_pt = v0_.cwiseMin(v1_).cwiseMin(v2_);
    Eigen::Vector3f max_pt = v0_.cwiseMax(v1_).cwiseMax(v2_);
    Eigen::Vector3f padding = Eigen::Vector3f::Constant(1e-4f);
    return AABB(min_pt - padding, max_pt + padding);
}
}

