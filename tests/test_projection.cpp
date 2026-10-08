#include "stage_projection.hpp"
#include "trace.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_reflection_on_hand_picked_vectors() {
    std::cout << "  Testing reflection on hand-picked vectors (Issue #13)..." << std::endl;
    mfad::Trace trace;

    // 1. Normal incidence
    Eigen::Vector3d v_normal(0.0, -1.0, 0.0);
    Eigen::Vector3d n(0.0, 1.0, 0.0);
    auto res_normal = mfad::stage_reflect(v_normal, n, &trace, "reflect_normal_incidence");
    assert(res_normal.success);
    assert((res_normal.value - Eigen::Vector3d(0.0, 1.0, 0.0)).norm() < 1e-12);

    // 2. 45-degree incidence
    Eigen::Vector3d v_45 = Eigen::Vector3d(1.0, -1.0, 0.0).normalized();
    auto res_45 = mfad::stage_reflect(v_45, n, &trace, "reflect_45_deg");
    assert(res_45.success);
    Eigen::Vector3d expected_45 = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
    assert((res_45.value - expected_45).norm() < 1e-12);

    // 3. Invariant checks in trace
    for (const auto& rec : trace.records()) {
        assert(rec.checks.at("length_preserved").get<bool>() == true);
        assert(rec.checks.at("angle_preserved").get<bool>() == true);
    }
}

void test_diffuse_term_on_hand_picked_vectors() {
    std::cout << "  Testing diffuse Lambertian term on hand-picked vectors (Issue #13)..."
              << std::endl;
    mfad::Trace trace;
    Eigen::Vector3d n(0.0, 1.0, 0.0);

    // 1. Directly overhead light (cos = 1.0)
    Eigen::Vector3d l_overhead(0.0, 1.0, 0.0);
    auto res_1 = mfad::stage_diffuse_term(n, l_overhead, &trace, "diffuse_overhead");
    assert(res_1.success);
    assert(std::abs(res_1.value - 1.0) < 1e-12);

    // 2. 60-degree light (cos(60 deg) = 0.5)
    Eigen::Vector3d l_60 = Eigen::Vector3d(std::sqrt(3.0) / 2.0, 0.5, 0.0);
    auto res_60 = mfad::stage_diffuse_term(n, l_60, &trace, "diffuse_60_deg");
    assert(res_60.success);
    assert(std::abs(res_60.value - 0.5) < 1e-12);

    // 3. 90-degree grazing light (cos = 0.0)
    Eigen::Vector3d l_grazing(1.0, 0.0, 0.0);
    auto res_grazing = mfad::stage_diffuse_term(n, l_grazing, &trace, "diffuse_grazing");
    assert(res_grazing.success);
    assert(std::abs(res_grazing.value - 0.0) < 1e-12);

    // 4. Backfacing light below surface (cos clamped to 0.0)
    Eigen::Vector3d l_back(0.0, -1.0, 0.0);
    auto res_back = mfad::stage_diffuse_term(n, l_back, &trace, "diffuse_backfacing");
    assert(res_back.success);
    assert(res_back.value == 0.0);

    assert(trace.records().size() == 4);
    assert(trace.records()[0].checks.at("illuminated").get<bool>() == true);
    assert(trace.records()[3].checks.at("illuminated").get<bool>() == false);
}

void test_shadow_direction_on_hand_picked_vectors() {
    std::cout << "  Testing shadow direction and ray offset test (Issue #13)..." << std::endl;
    mfad::Trace trace;

    Eigen::Vector3d surface_pt(1.0, 0.0, -2.0);
    Eigen::Vector3d light_pos(1.0, 4.0, -2.0);

    auto res =
        mfad::stage_shadow_direction(surface_pt, light_pos, 1e-4, &trace, "shadow_point_light");
    assert(res.success);
    assert(std::abs(res.value.distance - 4.0) < 1e-12);
    assert((res.value.direction - Eigen::Vector3d(0.0, 1.0, 0.0)).norm() < 1e-12);
    assert((res.value.shadow_origin - Eigen::Vector3d(1.0, 1e-4, -2.0)).norm() < 1e-12);

    assert(trace.records()[0].checks.at("valid_dist").get<bool>() == true);
    assert(trace.records()[0].checks.at("valid_dir").get<bool>() == true);
}

void test_projection_onto_image_plane() {
    std::cout << "  Testing projection onto image plane (Issue #13)..." << std::endl;
    mfad::Trace trace;

    const int width = 800;
    const int height = 600;
    const double vfov = 90.0;
    const double aspect = static_cast<double>(width) / static_cast<double>(height);

    // Center point along -Z in front of camera
    Eigen::Vector3d pt_center(0.0, 0.0, -5.0);
    auto res_center = mfad::stage_project_image_plane(pt_center, vfov, aspect, width, height, 0.1,
                                                      100.0, &trace, "project_center");
    assert(res_center.success);
    assert(res_center.value.in_frustum);
    // Center maps to NDC (0, 0)
    assert(std::abs(res_center.value.ndc.x()) < 1e-12);
    assert(std::abs(res_center.value.ndc.y()) < 1e-12);
    // Pixel is image center: px ~ (width-1)/2, py ~ (height-1)/2
    assert(std::abs(res_center.value.pixel.x() - 0.5 * (width - 1)) < 1e-10);
    assert(std::abs(res_center.value.pixel.y() - 0.5 * (height - 1)) < 1e-10);

    // Point outside frustum (behind camera)
    Eigen::Vector3d pt_behind(0.0, 0.0, 5.0);
    auto res_behind = mfad::stage_project_image_plane(pt_behind, vfov, aspect, width, height, 0.1,
                                                      100.0, &trace, "project_behind");
    assert(res_behind.success);
    assert(!res_behind.value.in_frustum);
}

int main() {
    std::cout << "[TEST] Running stage_projection verification tests (Issue #13)..." << std::endl;
    test_reflection_on_hand_picked_vectors();
    test_diffuse_term_on_hand_picked_vectors();
    test_shadow_direction_on_hand_picked_vectors();
    test_projection_onto_image_plane();
    std::cout << "[PASS] All stage_projection tests passed!" << std::endl;
    return 0;
}
