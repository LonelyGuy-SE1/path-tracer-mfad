#include "hit_record.hpp"
#include "material.hpp"
#include "ray.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_reflect() {
    std::cout << "  Testing reflect()..." << std::endl;
    // Normal incidence
    Eigen::Vector3f v_normal(0.0f, -1.0f, 0.0f);
    Eigen::Vector3f n(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f r1 = mfad::reflect(v_normal, n);
    assert((r1 - Eigen::Vector3f(0.0f, 1.0f, 0.0f)).norm() < 1e-6f);

    // 45-degree incidence
    Eigen::Vector3f v_45 = Eigen::Vector3f(1.0f, -1.0f, 0.0f).normalized();
    Eigen::Vector3f r2 = mfad::reflect(v_45, n);
    Eigen::Vector3f expected_r2 = Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized();
    assert((r2 - expected_r2).norm() < 1e-6f);

    // Magnitude preserved
    assert(std::abs(r2.norm() - 1.0f) < 1e-6f);
    // Angle of incidence equals angle of reflection
    assert(std::abs(v_45.dot(n) + r2.dot(n)) < 1e-6f);
}

void test_metal_scatter() {
    std::cout << "  Testing Metal material..." << std::endl;
    Eigen::Vector3f albedo(0.8f, 0.85f, 0.9f);
    mfad::Metal perfect_mirror(albedo, 0.0f);

    mfad::Ray r_in(Eigen::Vector3f(-1.0f, 1.0f, 0.0f),
                   Eigen::Vector3f(1.0f, -1.0f, 0.0f).normalized());
    mfad::HitRecord rec;
    rec.point = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    rec.normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    rec.front_face = true;

    mfad::ScatterRecord srec;
    bool scattered = perfect_mirror.scatter(r_in, rec, srec);
    assert(scattered);
    assert((srec.attenuation - albedo).norm() < 1e-6f);

    Eigen::Vector3f expected_dir = Eigen::Vector3f(1.0f, 1.0f, 0.0f).normalized();
    assert((srec.scattered.direction - expected_dir).norm() < 1e-5f);

    // Fuzzed mirror
    mfad::Metal fuzzed_mirror(albedo, 0.25f);
    int valid_bounces = 0;
    for (int i = 0; i < 100; ++i) {
        if (fuzzed_mirror.scatter(r_in, rec, srec)) {
            valid_bounces++;
            assert(srec.scattered.direction.dot(rec.normal) > 0.0f);
        }
    }
    assert(valid_bounces > 80);  // Most fuzzed rays remain in upper hemisphere
}

void test_refract_and_snell() {
    std::cout << "  Testing refract() and Snell's Law..." << std::endl;
    Eigen::Vector3f n(0.0f, 1.0f, 0.0f);

    // Normal incidence (theta_i = 0): ray continues straight through
    Eigen::Vector3f uv_normal(0.0f, -1.0f, 0.0f);
    Eigen::Vector3f refracted;
    bool ok = mfad::refract(uv_normal, n, 1.0f / 1.5f, refracted);
    assert(ok);
    assert((refracted - Eigen::Vector3f(0.0f, -1.0f, 0.0f)).norm() < 1e-6f);

    // Air (n=1.0) into Glass (n=1.5) at theta_i = 30 deg
    float theta_i = 30.0f * 3.14159265f / 180.0f;
    Eigen::Vector3f uv_30(std::sin(theta_i), -std::cos(theta_i), 0.0f);
    ok = mfad::refract(uv_30, n, 1.0f / 1.5f, refracted);
    assert(ok);

    // Snell's Law: sin(theta_t) = (1.0 / 1.5) * sin(30 deg) = 0.5 / 1.5 = 1/3
    float sin_theta_t = std::abs(refracted.x());
    assert(std::abs(sin_theta_t - (1.0f / 3.0f)) < 1e-5f);

    // Glass (n=1.5) into Air (n=1.0) with Total Internal Reflection (TIR)
    // Critical angle is arcsin(1.0 / 1.5) = 41.81 degrees.
    // Incident at 60 degrees from inside surface (normal pointing into air)
    float theta_tir = 60.0f * 3.14159265f / 180.0f;
    Eigen::Vector3f uv_tir(std::sin(theta_tir), -std::cos(theta_tir), 0.0f);
    bool tir_ok = mfad::refract(uv_tir, n, 1.5f / 1.0f, refracted);
    assert(!tir_ok);  // Total internal reflection must return false!
}

void test_schlick_reflectance() {
    std::cout << "  Testing Schlick reflectance approximation..." << std::endl;
    float ref_idx = 1.5f;

    // Normal incidence: R0 = ((1 - 1.5) / (1 + 1.5))^2 = (0.5 / 2.5)^2 = 0.04 (4%)
    float r_normal = mfad::reflectance(1.0f, ref_idx);
    assert(std::abs(r_normal - 0.04f) < 1e-5f);

    // Grazing incidence: R = 1.0 (100% reflection)
    float r_grazing = mfad::reflectance(0.0f, ref_idx);
    assert(std::abs(r_grazing - 1.0f) < 1e-5f);

    // Monotonicity: R(theta) decreases as cos(theta) increases
    float r_prev = 1.1f;
    for (float cos_theta = 0.0f; cos_theta <= 1.0f; cos_theta += 0.1f) {
        float r = mfad::reflectance(cos_theta, ref_idx);
        assert(r <= r_prev);
        r_prev = r;
    }
}

void test_dielectric_scatter() {
    std::cout << "  Testing Dielectric (Glass) scatter..." << std::endl;
    mfad::Dielectric glass(1.5f);
    assert(std::abs(glass.refraction_index() - 1.5f) < 1e-6f);

    // Total internal reflection test: ray exiting glass at 60 degrees
    mfad::Ray r_tir(Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                    Eigen::Vector3f(std::sin(60.0f * 3.14159265f / 180.0f),
                                    std::cos(60.0f * 3.14159265f / 180.0f), 0.0f));
    mfad::HitRecord rec_tir;
    rec_tir.point = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    rec_tir.normal = Eigen::Vector3f(0.0f, -1.0f, 0.0f);  // normal opposes ray inside glass
    rec_tir.front_face = false;                           // ray inside medium

    mfad::ScatterRecord srec;
    for (int i = 0; i < 50; ++i) {
        bool scattered = glass.scatter(r_tir, rec_tir, srec);
        assert(scattered);
        // Under TIR, every scatter must reflect, never refract
        assert(srec.scattered.direction.dot(rec_tir.normal) > 0.0f);
    }

    // Normal incidence: statistically ~4% reflection, ~96% refraction
    mfad::Ray r_normal(Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f(0.0f, -1.0f, 0.0f));
    mfad::HitRecord rec_normal;
    rec_normal.point = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    rec_normal.normal = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    rec_normal.front_face = true;

    int reflect_count = 0;
    int refract_count = 0;
    const int N = 10000;
    for (int i = 0; i < N; ++i) {
        glass.scatter(r_normal, rec_normal, srec);
        if (srec.scattered.direction.y() > 0.0f) {
            reflect_count++;
        } else {
            refract_count++;
        }
    }
    float refl_ratio = static_cast<float>(reflect_count) / static_cast<float>(N);
    // Should be close to theoretical 4% (0.04)
    assert(refl_ratio > 0.02f && refl_ratio < 0.06f);
    assert(refract_count > 9000);
}

int main() {
    std::cout << "[TEST] Running material verification tests (Issue #25)..." << std::endl;
    test_reflect();
    test_metal_scatter();
    test_refract_and_snell();
    test_schlick_reflectance();
    test_dielectric_scatter();
    std::cout << "[PASS] All material reflection, refraction, and Fresnel tests passed!"
              << std::endl;
    return 0;
}
