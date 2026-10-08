#include "quad.hpp"

#include <cmath>

namespace mfad {

Quad::Quad(const Eigen::Vector3f& Q, const Eigen::Vector3f& u, const Eigen::Vector3f& v,
           const Eigen::Vector3f& color, std::shared_ptr<Material> material)
    : Q_(Q), u_(u), v_(v), color_(color), material_(std::move(material)) {
    Eigen::Vector3f n_cross = u_.cross(v_);
    area_ = n_cross.norm();
    normal_ = n_cross.normalized();
    D_ = normal_.dot(Q_);
    w_ = n_cross / n_cross.squaredNorm();
}

bool Quad::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    float denom = normal_.dot(r.direction);
    // Ray parallel to plane
    if (std::abs(denom) < 1e-8f) {
        return false;
    }

    // Distance to plane
    float t = (D_ - normal_.dot(r.origin)) / denom;
    if (t < t_min || t > t_max) {
        return false;
    }

    // Intersection point on plane
    Eigen::Vector3f hit_pt = r.point_at(t);
    Eigen::Vector3f planar_hit = hit_pt - Q_;

    // Planar basis inversion via cross products:
    // alpha = w . (planar_hit x v)
    // beta  = w . (u x planar_hit)
    float alpha = w_.dot(planar_hit.cross(v_));
    float beta = w_.dot(u_.cross(planar_hit));

    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
        return false;
    }

    rec.t = t;
    rec.point = hit_pt;
    rec.set_face_normal(r, normal_);
    rec.color = color_;
    rec.material = material_.get();
    return true;
}

}  // namespace mfad
