#include "stage_interface.hpp"
#include "stage_transforms.hpp"
#include "trace.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>

void test_stage_interface_dummy() {
    std::cout << "  Testing Stage interface and dummy stage (Issue #4)..." << std::endl;
    mfad::Trace trace;
    mfad::StageResult<double> res = mfad::stage_dummy(5.0, trace, "dummy_test");

    assert(res.success);
    assert(std::abs(res.value - 10.0) < 1e-12);
    assert(res.stage_name == "dummy");
    assert(trace.records().size() == 1);
    assert(trace.records()[0].stage == "dummy");
    assert(trace.records()[0].name == "dummy_test");
}

void test_trace_binary_writer() {
    std::cout << "  Testing Trace writer binary blob output (Issue #5)..." << std::endl;
    // Set small inline limit of 100 elements to trigger binary blob storage
    mfad::Trace trace(100);

    // 15x15 matrix = 225 elements > 100 limit
    Eigen::MatrixXd big_mat = Eigen::MatrixXd::Random(15, 15);
    trace.record_matrix("test_stage", "big_matrix", big_mat, "Large matrix test");

    assert(trace.records().size() == 1);
    assert(trace.records()[0].data.is_object());
    assert(trace.records()[0].data.contains("$bin"));
    assert(trace.records()[0].data["$bin"]["nbytes"] == 225 * sizeof(double));

    const std::string json_file = "test_binary_trace.json";
    const std::string bin_file = "test_binary_trace.bin";
    bool saved = trace.save_json(json_file);
    assert(saved);

    // Verify binary file was written and matches size
    std::ifstream bin_in(bin_file, std::ios::binary | std::ios::ate);
    assert(bin_in.is_open());
    assert(static_cast<size_t>(bin_in.tellg()) == 225 * sizeof(double));
    bin_in.close();

    // Clean up test files
    std::remove(json_file.c_str());
    std::remove(bin_file.c_str());
}

void test_rotations() {
    std::cout << "  Testing 3D rotations and isometry invariants (Issue #10)..." << std::endl;
    mfad::Trace trace;
    const double pi = 3.14159265358979323846;

    // 1. Rotation about X
    Eigen::Matrix4d Rx = mfad::make_rotate_x(pi / 2.0);
    auto res_rx = mfad::stage_transforms(Rx, "rotate_x", &trace, "rot_x_90");
    assert(res_rx.success);
    // Rotating (0, 1, 0) about X by 90 deg -> (0, 0, 1)
    Eigen::Vector3d vy(0.0, 1.0, 0.0);
    Eigen::Vector3d vy_rot = mfad::transform_vector(Rx, vy);
    assert((vy_rot - Eigen::Vector3d(0.0, 0.0, 1.0)).norm() < 1e-12);

    // 2. Rotation about Y
    Eigen::Matrix4d Ry = mfad::make_rotate_y(pi / 2.0);
    auto res_ry = mfad::stage_transforms(Ry, "rotate_y", &trace, "rot_y_90");
    assert(res_ry.success);
    // Rotating (1, 0, 0) about Y by 90 deg -> (0, 0, -1)
    Eigen::Vector3d vx(1.0, 0.0, 0.0);
    Eigen::Vector3d vx_rot = mfad::transform_vector(Ry, vx);
    assert((vx_rot - Eigen::Vector3d(0.0, 0.0, -1.0)).norm() < 1e-12);

    // 3. Rotation about Z
    Eigen::Matrix4d Rz = mfad::make_rotate_z(pi / 2.0);
    auto res_rz = mfad::stage_transforms(Rz, "rotate_z", &trace, "rot_z_90");
    assert(res_rz.success);
    // Rotating (1, 0, 0) about Z by 90 deg -> (0, 1, 0)
    vx_rot = mfad::transform_vector(Rz, vx);
    assert((vx_rot - Eigen::Vector3d(0.0, 1.0, 0.0)).norm() < 1e-12);

    // 4. Rotation about arbitrary axis
    Eigen::Vector3d axis(1.0, 1.0, 1.0);
    Eigen::Matrix4d R_axis = mfad::make_rotate_axis(axis, 2.0 * pi / 3.0);
    auto res_ra = mfad::stage_transforms(R_axis, "rotate_axis", &trace, "rot_axis_120");
    assert(res_ra.success);
    // Rotating (1, 0, 0) about (1, 1, 1) by 120 deg -> (0, 1, 0)
    vx_rot = mfad::transform_vector(R_axis, vx);
    assert((vx_rot - Eigen::Vector3d(0.0, 1.0, 0.0)).norm() < 1e-12);

    // Verify rotation isometry checks in trace
    for (const auto& rec : trace.records()) {
        assert(rec.checks.at("QtQ_is_identity").get<bool>() == true);
        assert(rec.checks.at("det_is_one").get<bool>() == true);
        assert(rec.checks.at("passes_isometry").get<bool>() == true);
    }
}

void test_scale_and_shear_fail_isometry() {
    std::cout << "  Testing scale and shear fail isometry checks (Issue #10)..." << std::endl;
    mfad::Trace trace;

    // 1. Non-uniform scale
    Eigen::Matrix4d S_nonuniform = mfad::make_scale(Eigen::Vector3d(2.0, 1.0, 0.5));
    auto res_s = mfad::stage_transforms(S_nonuniform, "scale", &trace, "scale_nonuniform");
    assert(res_s.success);

    // 2. Uniform scale (scale = 2.0)
    Eigen::Matrix4d S_uniform = mfad::make_scale(2.0);
    auto res_su = mfad::stage_transforms(S_uniform, "scale", &trace, "scale_uniform");
    assert(res_su.success);

    // 3. Shear
    Eigen::Matrix4d Sh = mfad::make_shear(1.5, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto res_sh = mfad::stage_transforms(Sh, "shear", &trace, "shear_xy");
    assert(res_sh.success);

    // Verify that scale and shear correctly FAIL the isometry check!
    // Check scale_nonuniform
    assert(trace.records()[0].checks.at("passes_isometry").get<bool>() == false);
    // Check scale_uniform (det = 8 != 1)
    assert(trace.records()[1].checks.at("passes_isometry").get<bool>() == false);
    assert(std::abs(trace.records()[1].checks.at("det").get<double>() - 8.0) < 1e-12);
    // Check shear (QtQ != I)
    assert(trace.records()[2].checks.at("passes_isometry").get<bool>() == false);
    assert(trace.records()[2].checks.at("QtQ_is_identity").get<bool>() == false);
}

void test_composition_and_normal_transform() {
    std::cout << "  Testing 4x4 composition and normal transformation (Issue #10)..." << std::endl;
    const double pi = 3.14159265358979323846;

    Eigen::Matrix4d T = mfad::make_translate(Eigen::Vector3d(10.0, -5.0, 2.0));
    Eigen::Matrix4d R = mfad::make_rotate_z(pi / 2.0);
    Eigen::Matrix4d S = mfad::make_scale(Eigen::Vector3d(2.0, 3.0, 1.0));

    // Compose M = T * R * S
    Eigen::Matrix4d M = mfad::compose_transforms({S, R, T});

    // Test point transformation
    Eigen::Vector3d p(1.0, 0.0, 0.0);
    // S * p = (2, 0, 0)
    // R * (2, 0, 0) = (0, 2, 0)
    // T * (0, 2, 0) = (10, -3, 2)
    Eigen::Vector3d p_transformed = mfad::transform_point(M, p);
    assert((p_transformed - Eigen::Vector3d(10.0, -3.0, 2.0)).norm() < 1e-12);

    // Test surface normal transformation via inverse transpose
    // Surface normal on plane x=1 is (1, 0, 0)
    Eigen::Matrix4d S_nonuniform = mfad::make_scale(Eigen::Vector3d(2.0, 1.0, 1.0));
    Eigen::Vector3d n(1.0, 0.0, 0.0);
    Eigen::Vector3d n_transformed = mfad::transform_normal(S_nonuniform, n);
    // Normal should remain perpendicular to the stretched surface
    assert((n_transformed - Eigen::Vector3d(1.0, 0.0, 0.0)).norm() < 1e-12);
}

int main() {
    std::cout << "[TEST] Running stage_transforms, stage_interface, and trace_writer tests (Issue #4, #5, #10)..."
              << std::endl;
    test_stage_interface_dummy();
    test_trace_binary_writer();
    test_rotations();
    test_scale_and_shear_fail_isometry();
    test_composition_and_normal_transform();
    std::cout << "[PASS] All stage_transforms, stage_interface, and trace_writer tests passed!" << std::endl;
    return 0;
}
