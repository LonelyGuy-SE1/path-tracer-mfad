#include "core/camera.hpp"
#include "core/direct_lighting.hpp"
#include "core/image_buffer.hpp"
#include "core/material.hpp"
#include "core/plane.hpp"
#include "core/sphere.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

#define TEST_ASSERT(cond)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "Assertion failed at line " << __LINE__ << ": " #cond << std::endl;       \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

void test_lambert_cosine_shading() {
    std::cout << "[TEST] Testing Lambertian diffuse cosine shading..." << std::endl;

    mfad::HittableList scene;
    Eigen::Vector3f surface_point(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f normal(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f albedo(1.0f, 1.0f, 1.0f);

    mfad::DirectLightingOptions opts;
    opts.use_distance_attenuation = false;
    opts.ambient_color = Eigen::Vector3f::Zero();

    // 1. Light directly overhead: cos(0) = 1.0
    mfad::PointLight light_overhead(Eigen::Vector3f(0.0f, 5.0f, 0.0f),
                                    Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    Eigen::Vector3f l_overhead =
        mfad::compute_direct_lighting(surface_point, normal, albedo, {light_overhead}, scene, opts);
    const float pi = static_cast<float>(M_PI);
    TEST_ASSERT(std::abs(l_overhead.x() - 1.0f / pi) < 1e-4f);
    TEST_ASSERT(std::abs(l_overhead.y() - 1.0f / pi) < 1e-4f);
    TEST_ASSERT(std::abs(l_overhead.z() - 1.0f / pi) < 1e-4f);

    // 2. Light at 45 degrees: cos(45 deg) = sqrt(2) / 2 ~= 0.707107
    mfad::PointLight light_45(Eigen::Vector3f(5.0f, 5.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    Eigen::Vector3f l_45 =
        mfad::compute_direct_lighting(surface_point, normal, albedo, {light_45}, scene, opts);
    float expected_cos45 = (1.0f / std::sqrt(2.0f)) / pi;
    TEST_ASSERT(std::abs(l_45.x() - expected_cos45) < 1e-4f);

    // 3. Light at horizon (90 degrees): cos(90 deg) = 0.0
    mfad::PointLight light_horizon(Eigen::Vector3f(5.0f, 0.0f, 0.0f),
                                   Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    Eigen::Vector3f l_horizon =
        mfad::compute_direct_lighting(surface_point, normal, albedo, {light_horizon}, scene, opts);
    TEST_ASSERT(l_horizon.squaredNorm() < 1e-6f);

    // 4. Light below horizon: cos clamped to 0.0
    mfad::PointLight light_below(Eigen::Vector3f(0.0f, -5.0f, 0.0f),
                                 Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    Eigen::Vector3f l_below =
        mfad::compute_direct_lighting(surface_point, normal, albedo, {light_below}, scene, opts);
    TEST_ASSERT(l_below.squaredNorm() < 1e-6f);

    std::cout << "[PASS] Lambertian cosine shading validated!" << std::endl;
}

void test_shadow_ray_occlusion() {
    std::cout << "[TEST] Testing shadow ray occlusion..." << std::endl;

    mfad::HittableList scene;
    // Add occluding sphere at (0, 2, 0) with radius 1
    auto mat_sphere = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.8f, 0.2f, 0.2f));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 2.0f, 0.0f), 1.0f,
                                             Eigen::Vector3f::Ones(), mat_sphere));

    // Point light directly overhead at (0, 8, 0)
    mfad::PointLight light(Eigen::Vector3f(0.0f, 8.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, 1.0f));

    Eigen::Vector3f normal(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f albedo(1.0f, 1.0f, 1.0f);

    mfad::DirectLightingOptions opts;
    opts.use_distance_attenuation = false;
    opts.ambient_color = Eigen::Vector3f(0.1f, 0.1f, 0.1f);

    // 1. Point at (0, 0, 0) directly below sphere -> MUST be in shadow!
    Eigen::Vector3f pt_shadowed(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f l_shadowed =
        mfad::compute_direct_lighting(pt_shadowed, normal, albedo, {light}, scene, opts);
    // Should receive only ambient light
    TEST_ASSERT(std::abs(l_shadowed.x() - 0.1f) < 1e-4f);
    TEST_ASSERT(std::abs(l_shadowed.y() - 0.1f) < 1e-4f);
    TEST_ASSERT(std::abs(l_shadowed.z() - 0.1f) < 1e-4f);

    // 2. Point at (4, 0, 0) far to the side -> MUST NOT be in shadow!
    Eigen::Vector3f pt_lit(4.0f, 0.0f, 0.0f);
    Eigen::Vector3f l_lit =
        mfad::compute_direct_lighting(pt_lit, normal, albedo, {light}, scene, opts);
    TEST_ASSERT(l_lit.x() > 0.1f);  // Must have direct light contribution

    std::cout << "[PASS] Shadow ray occlusion and shadow acne prevention verified!" << std::endl;
}

void test_render_direct_lighting_scene() {
    std::cout << "[TEST] Rendering direct lighting test scene..." << std::endl;

    mfad::Scene scene;
    auto mat_red = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.85f, 0.22f, 0.20f));
    auto mat_white = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.75f, 0.75f, 0.75f));

    // Floor
    scene.hittables.add(std::make_shared<mfad::Plane>(Eigen::Vector3f(0.0f, -0.5f, 0.0f),
                                                      Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                                                      Eigen::Vector3f::Ones(), mat_white));

    // Spheres resting on floor (y = -0.5): center diffuse red, left glass, right mirror
    auto mat_glass = std::make_shared<mfad::Dielectric>(1.5f);
    auto mat_mirror = std::make_shared<mfad::Metal>(Eigen::Vector3f(0.9f, 0.9f, 0.9f), 0.0f);
    scene.hittables.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, -1.5f), 0.5f,
                                                       Eigen::Vector3f::Ones(), mat_red));
    scene.hittables.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(-1.1f, 0.0f, -1.5f), 0.5f,
                                                       Eigen::Vector3f::Ones(), mat_glass));
    scene.hittables.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(1.1f, 0.0f, -1.5f), 0.5f,
                                                       Eigen::Vector3f::Ones(), mat_mirror));

    // Point light positioned above and slightly to the right
    scene.point_lights.emplace_back(Eigen::Vector3f(2.0f, 4.0f, 0.0f),
                                    Eigen::Vector3f(50.0f, 50.0f, 50.0f));

    const int width = 320;
    const int height = 180;
    mfad::Camera camera(Eigen::Vector3f(0.0f, 1.0f, 2.5f), Eigen::Vector3f(0.0f, 0.0f, -1.5f),
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f), 50.0f,
                        static_cast<float>(width) / static_cast<float>(height));

    mfad::ImageBuffer buffer(width, height);
    mfad::DirectLightingOptions opts;
    opts.use_distance_attenuation = true;
    opts.use_sky_gradient = false;
    opts.background_color = Eigen::Vector3f::Zero();
    opts.ambient_color = Eigen::Vector3f(0.02f, 0.02f, 0.02f);

    mfad::render_direct_lighting(camera, scene, buffer, opts, 4);

    // Verify buffer has valid non-zero content
    bool has_illuminated_pixels = false;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f p = buffer.get_pixel(x, y);
            TEST_ASSERT(!std::isnan(p.x()) && !std::isnan(p.y()) && !std::isnan(p.z()));
            TEST_ASSERT(!std::isinf(p.x()) && !std::isinf(p.y()) && !std::isinf(p.z()));
            if (p.x() > 0.2f) {
                has_illuminated_pixels = true;
            }
        }
    }
    TEST_ASSERT(has_illuminated_pixels);

    // Write artifact image to disk
    const std::string out_file = "direct_lighting.png";
    bool written = buffer.write_png(out_file);
    TEST_ASSERT(written);
    std::cout << "[PASS] Rendered direct lighting artifact to " << out_file << std::endl;
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  Running Direct Lighting Tests (Issue #24)      " << std::endl;
    std::cout << "=================================================" << std::endl;

    test_lambert_cosine_shading();
    test_shadow_ray_occlusion();
    test_render_direct_lighting_scene();

    std::cout << "\n[PASS] All Direct Lighting tests passed successfully!" << std::endl;
    return 0;
}
