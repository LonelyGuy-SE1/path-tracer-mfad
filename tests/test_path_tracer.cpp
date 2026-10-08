#include "hittable.hpp"
#include "material.hpp"
#include "path_tracer.hpp"
#include "quad.hpp"
#include "sphere.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_quad_intersections() {
    std::cout << "  Testing Quad primitive intersections..." << std::endl;
    // 2x2 Quad on XY plane at Z = -1, corner at (-1, -1, -1), u = (2, 0, 0), v = (0, 2, 0)
    Eigen::Vector3f Q(-1.0f, -1.0f, -1.0f);
    Eigen::Vector3f u(2.0f, 0.0f, 0.0f);
    Eigen::Vector3f v(0.0f, 2.0f, 0.0f);
    auto mat = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.8f, 0.8f, 0.8f));
    mfad::Quad quad(Q, u, v, Eigen::Vector3f::Ones(), mat);

    // 1. Ray hitting center of quad
    mfad::Ray r_center(Eigen::Vector3f(0.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    mfad::HitRecord rec;
    bool hit = quad.hit(r_center, 0.001f, 100.0f, rec);
    assert(hit);
    assert(std::abs(rec.t - 2.0f) < 1e-5f);
    assert((rec.point - Eigen::Vector3f(0.0f, 0.0f, -1.0f)).norm() < 1e-5f);
    assert(rec.front_face);
    assert((rec.normal - Eigen::Vector3f(0.0f, 0.0f, 1.0f)).norm() < 1e-5f);

    // 2. Ray missing quad (outside bounds: x = 2.0)
    mfad::Ray r_miss(Eigen::Vector3f(2.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    assert(!quad.hit(r_miss, 0.001f, 100.0f, rec));

    // 3. Ray parallel to quad plane
    mfad::Ray r_parallel(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(1.0f, 0.0f, 0.0f));
    assert(!quad.hit(r_parallel, 0.001f, 100.0f, rec));

    // 4. Ray pointing away from quad
    mfad::Ray r_away(Eigen::Vector3f(0.0f, 0.0f, 1.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    assert(!quad.hit(r_away, 0.001f, 100.0f, rec));
}

void test_direct_emission() {
    std::cout << "  Testing direct emission from light sources..." << std::endl;
    mfad::HittableList scene;
    Eigen::Vector3f emit_color(10.0f, 5.0f, 2.0f);
    auto light_mat = std::make_shared<mfad::DiffuseLight>(emit_color);
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, -2.0f), 1.0f,
                                             Eigen::Vector3f::Ones(), light_mat));

    mfad::PathTracerOptions opts;
    opts.max_bounces = 8;
    mfad::PathTracer tracer(opts);

    // Ray directed straight at the light sphere
    mfad::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    Eigen::Vector3f L = tracer.trace_ray(r, scene);
    assert((L - emit_color).norm() < 1e-5f);

    // Ray missing the light in void
    mfad::Ray r_miss(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    Eigen::Vector3f L_miss = tracer.trace_ray(r_miss, scene);
    assert(L_miss.norm() < 1e-6f);
}

void test_furnace_energy_conservation() {
    std::cout << "  Testing Furnace test (energy conservation equilibrium)..." << std::endl;
    // Enclosed furnace: an inward-facing cavity with uniform diffuse emission Le
    // By the 2nd law of thermodynamics, radiance inside an isothermal cavity
    // and reflected by any conservative (albedo = 1.0) surface equals Le!
    mfad::HittableList scene;
    Eigen::Vector3f emit_val(2.0f, 2.0f, 2.0f);
    auto furnace_mat = std::make_shared<mfad::DiffuseLight>(emit_val, /*two_sided=*/true);
    // Large sphere enclosing the scene
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, 0.0f), 10.0f,
                                             Eigen::Vector3f::Ones(), furnace_mat));

    // Conservative white diffuse sphere (albedo = 1.0) inside the furnace cavity
    auto white_diffuse = std::make_shared<mfad::Lambertian>(Eigen::Vector3f::Ones());
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, -3.0f), 1.0f,
                                             Eigen::Vector3f::Ones(), white_diffuse));

    mfad::PathTracerOptions opts;
    opts.max_bounces = 16;
    opts.min_rr_bounces = 3;
    mfad::PathTracer tracer(opts);

    // 1. Direct rays from origin hitting furnace wall
    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 20; ++i) {
        Eigen::Vector3f dir(dist(rng), dist(rng), dist(rng));
        mfad::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), dir.normalized());
        Eigen::Vector3f L = tracer.trace_ray(r, scene);
        assert((L - emit_val).norm() < 1e-4f);
    }

    // 2. Rays hitting the white diffuse sphere inside furnace (must also reflect exactly Le)
    mfad::Ray r_sphere(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    Eigen::Vector3f L_sphere = tracer.trace_ray(r_sphere, scene);
    assert((L_sphere - emit_val).norm() < 1e-4f);
}

void test_russian_roulette_unbiasedness() {
    std::cout << "  Testing Russian roulette statistical unbiasedness..." << std::endl;
    // Two parallel plates: floor diffuse with albedo 0.5, ceiling emitter with Le = 10.0
    // At each bounce throughput is scaled by 0.5. After 2 bounces RR kicks in.
    // The average radiance across samples must converge to the true value without bias.
    mfad::HittableList scene;
    auto mat_light = std::make_shared<mfad::DiffuseLight>(Eigen::Vector3f(10.0f, 10.0f, 10.0f));
    auto mat_diffuse = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.5f, 0.5f, 0.5f));

    // Ceiling light at y = 1, pointing down (u x v = (10, 0, 0) x (0, 0, 10) = (0, -100, 0))
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-5.0f, 1.0f, -5.0f), Eigen::Vector3f(10.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 10.0f), Eigen::Vector3f::Ones(), mat_light));

    // Floor diffuse at y = -1, pointing up (u x v = (10, 0, 0) x (0, 0, -10) = (0, 100, 0))
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-5.0f, -1.0f, 5.0f), Eigen::Vector3f(10.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, -10.0f), Eigen::Vector3f::Ones(), mat_diffuse));

    mfad::PathTracerOptions opts;
    opts.max_bounces = 32;
    opts.min_rr_bounces = 2;
    mfad::PathTracer tracer(opts);

    // Ray hitting floor directly
    mfad::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, -1.0f, 0.0f));

    Eigen::Vector3f sum_L = Eigen::Vector3f::Zero();
    const int N = 2000;
    for (int i = 0; i < N; ++i) {
        sum_L += tracer.trace_ray(r, scene);
    }
    Eigen::Vector3f mean_L = sum_L / static_cast<float>(N);

    // Bounces off floor (albedo 0.5) directly to ceiling (10.0) gives first bounce contribution
    // ~ 5.0. Plus subsequent bounces back and forth: 5.0 / (1 - 0.5 * 0) = 5.0 Mean should be
    // around 5.0
    assert(mean_L.x() > 3.0f && mean_L.x() < 7.0f);
}

void test_area_light_sampling_soft_shadows() {
    std::cout << "  Testing area light direct sampling and soft shadows (Issue #27)..."
              << std::endl;
    mfad::HittableList scene;

    // Diffuse floor at y = 0
    auto floor_mat = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.8f, 0.8f, 0.8f));
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-10.0f, 0.0f, 10.0f), Eigen::Vector3f(20.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, -20.0f), Eigen::Vector3f::Ones(), floor_mat));

    // Area light at y = 4, size 2x2
    auto light_mat = std::make_shared<mfad::DiffuseLight>(Eigen::Vector3f(20.0f, 20.0f, 20.0f),
                                                          /*two_sided=*/true);
    auto quad_light = std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-1.0f, 4.0f, -1.0f), Eigen::Vector3f(2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 2.0f), Eigen::Vector3f::Ones(), light_mat);
    scene.add(quad_light);

    // Occluding sphere at (0, 2, 0), radius 0.5
    auto blocker_mat = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.1f, 0.1f, 0.1f));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 2.0f, 0.0f), 0.5f,
                                             Eigen::Vector3f::Ones(), blocker_mat));

    mfad::PathTracerOptions opts;
    opts.sample_lights = true;
    opts.area_lights.push_back(quad_light);
    opts.max_bounces = 4;
    mfad::PathTracer tracer(opts);

    // Ray looking at umbra (point on floor at x=0, y=0, z=0 directly below blocker)
    mfad::Ray r_umbra(Eigen::Vector3f(0.0f, 1.0f, 2.0f),
                      Eigen::Vector3f(0.0f, -1.0f, -2.0f).normalized());
    // Ray looking at penumbra (point on floor near shadow edge at x=0.4, y=0, z=0)
    mfad::Ray r_penumbra(Eigen::Vector3f(0.4f, 1.0f, 2.0f),
                         Eigen::Vector3f(0.0f, -1.0f, -2.0f).normalized());
    // Ray looking at unoccluded floor just outside shadow at x=1.0, y=0, z=0
    mfad::Ray r_lit(Eigen::Vector3f(1.0f, 1.0f, 2.0f),
                    Eigen::Vector3f(0.0f, -1.0f, -2.0f).normalized());

    Eigen::Vector3f L_umbra = Eigen::Vector3f::Zero();
    Eigen::Vector3f L_penumbra = Eigen::Vector3f::Zero();
    Eigen::Vector3f L_lit = Eigen::Vector3f::Zero();

    const int N = 100;
    for (int i = 0; i < N; ++i) {
        L_umbra += tracer.trace_ray(r_umbra, scene);
        L_penumbra += tracer.trace_ray(r_penumbra, scene);
        L_lit += tracer.trace_ray(r_lit, scene);
    }
    L_umbra /= static_cast<float>(N);
    L_penumbra /= static_cast<float>(N);
    L_lit /= static_cast<float>(N);

    // Umbra should be darker than penumbra, and penumbra darker than fully lit
    assert(L_umbra.x() < L_penumbra.x());
    assert(L_penumbra.x() < L_lit.x());
}

int main() {
    std::cout << "[TEST] Running PathTracer & Russian Roulette verification tests (Issue #26)..."
              << std::endl;
    test_quad_intersections();
    test_direct_emission();
    test_furnace_energy_conservation();
    test_russian_roulette_unbiasedness();
    test_area_light_sampling_soft_shadows();
    std::cout << "[PASS] All PathTracer, Quad, and Russian Roulette tests passed!" << std::endl;
    return 0;
}
