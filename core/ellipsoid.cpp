#include "ellipsoid.hpp"

#include <cmath>

namespace mfad {

Ellipsoid::Ellipsoid(const Eigen::Vector3f& center, const Eigen::Vector3f& radii,
                     const Eigen::Matrix3f& rotation, const Eigen::Vector3f& color,
                     std::shared_ptr<Material> material)
    : center_(center), radii_(radii), rotation_(rotation), color_(color),
      material_(std::move(material)), raw_material_(material_ ? material_.get() : nullptr) {
    // M^-1 = diag(1/r) * R^T
    Eigen::Matrix3f inv_scale = Eigen::Matrix3f::Zero();
    inv_scale(0, 0) = 1.0f / radii_.x();
    inv_scale(1, 1) = 1.0f / radii_.y();
    inv_scale(2, 2) = 1.0f / radii_.z();
    inv_rotation_scale_ = inv_scale * rotation_.transpose();

    // A = R * diag(1/r^2) * R^T
    Eigen::Matrix3f inv_scale_sq = Eigen::Matrix3f::Zero();
    inv_scale_sq(0, 0) = 1.0f / (radii_.x() * radii_.x());
    inv_scale_sq(1, 1) = 1.0f / (radii_.y() * radii_.y());
    inv_scale_sq(2, 2) = 1.0f / (radii_.z() * radii_.z());
    a_mat_ = rotation_ * inv_scale_sq * rotation_.transpose();

    // Construct 4x4 algebraic quadric matrix in double precision
    Eigen::Matrix3d rot_d = rotation_.cast<double>();
    Eigen::Vector3d radii_d = radii_.cast<double>();
    Eigen::Vector3d center_d = center_.cast<double>();

    Eigen::Matrix3d inv_sq_d = Eigen::Matrix3d::Zero();
    inv_sq_d(0, 0) = 1.0 / (radii_d.x() * radii_d.x());
    inv_sq_d(1, 1) = 1.0 / (radii_d.y() * radii_d.y());
    inv_sq_d(2, 2) = 1.0 / (radii_d.z() * radii_d.z());
    Eigen::Matrix3d a_d = rot_d * inv_sq_d * rot_d.transpose();
    Eigen::Vector3d b_d = -a_d * center_d;
    double d_val = center_d.dot(a_d * center_d) - 1.0;

    quadric_matrix_ = Eigen::Matrix4d::Zero();
    quadric_matrix_.block<3, 3>(0, 0) = a_d;
    quadric_matrix_.block<3, 1>(0, 3) = b_d;
    quadric_matrix_.block<1, 3>(3, 0) = b_d.transpose();
    quadric_matrix_(3, 3) = d_val;
}

bool Ellipsoid::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    // Transform ray into unit-sphere space: ou = M^-1 * (origin - center), du = M^-1 * direction
    Eigen::Vector3f ou = inv_rotation_scale_ * (r.origin - center_);
    Eigen::Vector3f du = inv_rotation_scale_ * r.direction;

    float a = du.squaredNorm();
    float b = ou.dot(du);
    float c = ou.squaredNorm() - 1.0f;
    float discriminant = b * b - a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    float sqrtd = std::sqrt(discriminant);
    float root = (-b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.point = r.at(rec.t);
    // World space normal from gradient of implicit equation: grad F(p) = 2 * A * (p - c)
    Eigen::Vector3f outward_normal = (a_mat_ * (rec.point - center_)).normalized();
    rec.color = color_;
    rec.material = raw_material_;
    rec.set_face_normal(r, outward_normal);
    return true;
}

}  // namespace mfad
