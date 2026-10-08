#pragma once

#include "hit_record.hpp"
#include "ray.hpp"
#include "stage_basis.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

namespace mfad {

struct ScatterRecord {
    Ray scattered;
    Eigen::Vector3f attenuation{1.0f, 1.0f, 1.0f};
};

/**
 * @brief Thread-safe fast uniform random float generator in [0.0, 1.0).
 */
inline float random_float() {
    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

/**
 * @brief Generates a random vector inside the unit sphere for rough metal scattering.
 */
inline Eigen::Vector3f random_in_unit_sphere() {
    while (true) {
        Eigen::Vector3f p(2.0f * random_float() - 1.0f, 2.0f * random_float() - 1.0f,
                          2.0f * random_float() - 1.0f);
        if (p.squaredNorm() < 1.0f) {
            return p;
        }
    }
}

/**
 * @brief Base Material interface.
 */
class Material {
public:
    virtual ~Material() = default;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const = 0;
    virtual Eigen::Vector3f emitted(const Ray& /*r_in*/, const HitRecord& /*rec*/) const {
        return Eigen::Vector3f::Zero();
    }
};

/**
 * @brief Reflects vector v across unit normal n using orthogonal projection:
 * r = v - 2 * (v . n) * n
 */
inline Eigen::Vector3f reflect(const Eigen::Vector3f& v, const Eigen::Vector3f& n) {
    return v - 2.0f * v.dot(n) * n;
}

/**
 * @brief Refracts vector uv through normal n using Snell's Law.
 */
inline bool refract(const Eigen::Vector3f& uv, const Eigen::Vector3f& n, float etai_over_etat,
                    Eigen::Vector3f& refracted) {
    float cos_theta = std::min(-uv.dot(n), 1.0f);
    Eigen::Vector3f r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float k = 1.0f - r_out_perp.squaredNorm();
    if (k < 0.0f) {
        return false;  // Total internal reflection
    }
    Eigen::Vector3f r_out_parallel = -std::sqrt(k) * n;
    refracted = r_out_perp + r_out_parallel;
    return true;
}

/**
 * @brief Schlick's approximation for Fresnel reflectance.
 */
inline float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow(1.0f - cosine, 5.0f);
}

/**
 * @brief Cosine-weighted hemispherical sampling around a surface normal using Duff et al.
 * orthonormal basis.
 */
inline Eigen::Vector3f sample_cosine_hemisphere(const Eigen::Vector3f& normal) {
    const float pi = 3.14159265358979323846f;
    float r1 = random_float();
    float r2 = random_float();
    float phi = 2.0f * pi * r1;
    float r = std::sqrt(r2);
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    float z = std::sqrt(std::max(0.0f, 1.0f - r2));

    BasisResult basis = stage_basis_normal(normal.cast<double>());
    return basis.to_world(Eigen::Vector3d(x, y, z)).cast<float>().normalized();
}

/**
 * @brief Diffuse (Lambertian) Material using stage_basis for cosine-weighted bounce sampling.
 */
class Lambertian : public Material {
public:
    explicit Lambertian(const Eigen::Vector3f& albedo) : albedo_(albedo) {}

    bool scatter(const Ray& /*r_in*/, const HitRecord& rec, ScatterRecord& srec) const override {
        Eigen::Vector3f scatter_dir = sample_cosine_hemisphere(rec.normal);
        srec.scattered = Ray(rec.point + 1e-4f * rec.normal, scatter_dir);
        srec.attenuation = albedo_;
        return true;
    }

    const Eigen::Vector3f& albedo() const { return albedo_; }

private:
    Eigen::Vector3f albedo_;
};

/**
 * @brief Procedural Checkerboard Material for planes and floors.
 */
class CheckerMaterial : public Material {
public:
    CheckerMaterial(const Eigen::Vector3f& c1, const Eigen::Vector3f& c2, float scale = 2.0f)
        : color1_(c1), color2_(c2), scale_(scale) {}

    bool scatter(const Ray& /*r_in*/, const HitRecord& rec, ScatterRecord& srec) const override {
        int cx = static_cast<int>(std::floor(rec.point.x() * scale_));
        int cz = static_cast<int>(std::floor(rec.point.z() * scale_));
        bool is_even = ((cx + cz) % 2 + 2) % 2 == 0;
        Eigen::Vector3f albedo = is_even ? color1_ : color2_;

        Eigen::Vector3f scatter_dir = sample_cosine_hemisphere(rec.normal);
        srec.scattered = Ray(rec.point + 1e-4f * rec.normal, scatter_dir);
        srec.attenuation = albedo;
        return true;
    }

private:
    Eigen::Vector3f color1_;
    Eigen::Vector3f color2_;
    float scale_;
};

/**
 * @brief Metal (Mirror) Material with orthogonal reflection and optional roughness fuzz.
 */
class Metal : public Material {
public:
    explicit Metal(const Eigen::Vector3f& albedo, float fuzz = 0.0f)
        : albedo_(albedo), fuzz_(std::clamp(fuzz, 0.0f, 1.0f)) {}

    bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
        Eigen::Vector3f reflected = reflect(r_in.direction.normalized(), rec.normal);
        Eigen::Vector3f scattered_dir = reflected;
        if (fuzz_ > 0.0f) {
            scattered_dir = (reflected + fuzz_ * random_in_unit_sphere()).normalized();
        }
        srec.scattered = Ray(rec.point + 1e-4f * rec.normal, scattered_dir);
        srec.attenuation = albedo_;
        return srec.scattered.direction.dot(rec.normal) > 0.0f;
    }

private:
    Eigen::Vector3f albedo_;
    float fuzz_;
};

/**
 * @brief Glass (Dielectric) Material with Snell's law refraction & random Fresnel choice.
 */
class Dielectric : public Material {
public:
    explicit Dielectric(float refraction_index) : ir_(refraction_index) {}

    bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
        srec.attenuation = Eigen::Vector3f(1.0f, 1.0f, 1.0f);  // Pure glass does not absorb
        float refraction_ratio = rec.front_face ? (1.0f / ir_) : ir_;

        Eigen::Vector3f unit_dir = r_in.direction.normalized();
        float cos_theta = std::min(-unit_dir.dot(rec.normal), 1.0f);
        float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

        bool cannot_refract = (refraction_ratio * sin_theta > 1.0f);
        Eigen::Vector3f direction;

        float refl_prob = reflectance(cos_theta, refraction_ratio);

        // Random Fresnel choice: reflect with probability refl_prob, otherwise refract
        if (cannot_refract || refl_prob > random_float()) {
            direction = reflect(unit_dir, rec.normal);
        } else {
            if (!refract(unit_dir, rec.normal, refraction_ratio, direction)) {
                direction = reflect(unit_dir, rec.normal);
            }
        }

        Eigen::Vector3f offset_dir = (direction.dot(rec.normal) > 0.0f) ? rec.normal : -rec.normal;
        srec.scattered = Ray(rec.point + 1e-4f * offset_dir, direction);
        return true;
    }

    float refraction_index() const { return ir_; }

private:
    float ir_;
};

/**
 * @brief Diffuse emissive light material (area light).
 */
class DiffuseLight : public Material {
public:
    explicit DiffuseLight(const Eigen::Vector3f& emit, bool two_sided = false)
        : emit_(emit), two_sided_(two_sided) {}

    bool scatter(const Ray& /*r_in*/, const HitRecord& /*rec*/,
                 ScatterRecord& /*srec*/) const override {
        return false;
    }

    Eigen::Vector3f emitted(const Ray& /*r_in*/, const HitRecord& rec) const override {
        // Emit if two-sided or if ray strikes the front face
        if (two_sided_ || rec.front_face) {
            return emit_;
        }
        return Eigen::Vector3f::Zero();
    }

    bool two_sided() const { return two_sided_; }
    const Eigen::Vector3f& emit() const { return emit_; }

private:
    Eigen::Vector3f emit_;
    bool two_sided_{false};
};

}  // namespace mfad
