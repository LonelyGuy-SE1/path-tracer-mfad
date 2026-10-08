#include "hit_record.hpp"
#include "plane.hpp"
#include "ray.hpp"
#include "sphere.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_sphere_intersection() {
    mfad::Sphere sphere(Eigen::Vector3f(0.0f, 0.0f, -3.0f), 1.0f);

    // 1. Ray through sphere center
    mfad::Ray ray_hit(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    mfad::HitRecord rec;
    bool hit = sphere.hit(ray_hit, 0.001f, 100.0f, rec);
    assert(hit);
    assert(std::abs(rec.t - 2.0f) < 1e-5f);
    assert((rec.point - Eigen::Vector3f(0.0f, 0.0f, -2.0f)).norm() < 1e-5f);
    assert((rec.normal - Eigen::Vector3f(0.0f, 0.0f, 1.0f)).norm() < 1e-5f);

    // 2. Ray missing sphere
    mfad::Ray ray_miss(Eigen::Vector3f(0.0f, 2.5f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    hit = sphere.hit(ray_miss, 0.001f, 100.0f, rec);
    assert(!hit);

    // 3. Ray pointing away from sphere
    mfad::Ray ray_away(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    hit = sphere.hit(ray_away, 0.001f, 100.0f, rec);
    assert(!hit);
}

void test_plane_intersection() {
    mfad::Plane plane(Eigen::Vector3f(0.0f, -1.0f, 0.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f));

    // 1. Ray pointing down at plane
    mfad::Ray ray_hit(Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f(0.0f, -1.0f, 0.0f));
    mfad::HitRecord rec;
    bool hit = plane.hit(ray_hit, 0.001f, 100.0f, rec);
    assert(hit);
    assert(std::abs(rec.t - 3.0f) < 1e-5f);
    assert(std::abs(rec.point.y() - (-1.0f)) < 1e-5f);
    assert(rec.normal.y() == 1.0f);

    // 2. Ray parallel to plane
    mfad::Ray ray_parallel(Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f(1.0f, 0.0f, 0.0f));
    hit = plane.hit(ray_parallel, 0.001f, 100.0f, rec);
    assert(!hit);

    // 3. Ray pointing away from plane
    mfad::Ray ray_away(Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    hit = plane.hit(ray_away, 0.001f, 100.0f, rec);
    assert(!hit);
}

int main() {
    std::cout << "[TEST] Running sphere and plane intersection tests..." << std::endl;
    test_sphere_intersection();
    test_plane_intersection();
    std::cout << "[PASS] All sphere and plane intersection unit tests passed!" << std::endl;
    return 0;
}
