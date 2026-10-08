#include "stage_basis.hpp"
#include "trace.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

void test_basis_invariants(const Eigen::Vector3d& normal, mfad::Trace& trace, int index) {
    auto res_duff =
        mfad::stage_basis_normal(normal, &trace, "duff_sample_" + std::to_string(index));

    // 1. Check columns are unit length
    assert(std::abs(res_duff.tangent.norm() - 1.0) < 1e-12);
    assert(std::abs(res_duff.bitangent.norm() - 1.0) < 1e-12);
    assert(std::abs(res_duff.normal.norm() - 1.0) < 1e-12);

    // 2. Check normal equals 3rd column of Q
    assert((res_duff.Q.col(2) - res_duff.normal).cwiseAbs().maxCoeff() < 1e-12);

    // 3. Check orthogonality: Q^T * Q = I
    Eigen::Matrix3d QtQ = res_duff.Q.transpose() * res_duff.Q;
    double QtQ_err = (QtQ - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    assert(QtQ_err < 1e-12);

    // 4. Check right-handed orientation: det(Q) = +1
    double det = res_duff.Q.determinant();
    assert(std::abs(det - 1.0) < 1e-12);

    // 5. Check round-trip coordinate transformation
    Eigen::Vector3d v_local(0.123, -0.456, 0.789);
    Eigen::Vector3d v_world = res_duff.to_world(v_local);
    Eigen::Vector3d v_back = res_duff.to_local(v_world);
    double roundtrip_err = (v_back - v_local).cwiseAbs().maxCoeff();
    assert(roundtrip_err < 1e-12);

    // Test Gram-Schmidt as well
    Eigen::Vector3d guide(0.0, 1.0, 0.0);
    auto res_gs =
        mfad::stage_basis_gram_schmidt(normal, guide, &trace, "gs_sample_" + std::to_string(index));
    Eigen::Matrix3d QtQ_gs = res_gs.Q.transpose() * res_gs.Q;
    assert((QtQ_gs - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 1e-12);
    assert(std::abs(res_gs.Q.determinant() - 1.0) < 1e-12);
}

void test_camera_basis(const Eigen::Vector3d& look_from, const Eigen::Vector3d& look_at,
                       const Eigen::Vector3d& up, mfad::Trace& trace, const std::string& name) {
    auto res = mfad::stage_basis_camera(look_from, look_at, up, &trace, name);

    // 1. Check unit lengths
    assert(std::abs(res.right().norm() - 1.0) < 1e-12);
    assert(std::abs(res.up().norm() - 1.0) < 1e-12);
    assert(std::abs(res.forward().norm() - 1.0) < 1e-12);

    // 2. Check columns match semantic axes
    assert((res.Q.col(0) - res.right()).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.Q.col(1) - res.up()).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.Q.col(2) - res.forward()).cwiseAbs().maxCoeff() < 1e-12);

    // 3. Check orthogonality: Q^T * Q = I
    Eigen::Matrix3d QtQ = res.Q.transpose() * res.Q;
    double QtQ_err = (QtQ - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
    assert(QtQ_err < 1e-12);

    // 4. Check right-handed orientation: det(Q) = +1
    double det = res.Q.determinant();
    assert(std::abs(det - 1.0) < 1e-12);

    // 5. Check round-trip coordinate transformation
    Eigen::Vector3d v_local(0.333, -0.666, 0.999);
    Eigen::Vector3d v_world = res.to_world(v_local);
    Eigen::Vector3d v_back = res.to_local(v_world);
    assert((v_back - v_local).cwiseAbs().maxCoeff() < 1e-12);
}

int main() {
    std::cout << "[TEST] Running stage_basis verification tests..." << std::endl;
    mfad::Trace trace;

    // Fixed cardinal & edge cases for stage_basis_normal
    std::vector<Eigen::Vector3d> edge_normals = {
        {0, 0, 1},
        {0, 0, -1},
        {1, 0, 0},
        {-1, 0, 0},
        {0, 1, 0},
        {0, -1, 0},
        {0, 0, 0.5},  // tests auto-normalization
        {1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)},
        {0.0001, -0.0001, 0.999999}};

    int idx = 0;
    for (const auto& n : edge_normals) {
        test_basis_invariants(n, trace, idx++);
    }

    // 1000 Random spherical directions
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
        Eigen::Vector3d rand_n(dist(rng), dist(rng), dist(rng));
        test_basis_invariants(rand_n, trace, idx++);
    }

    // Test Issue #11 camera basis
    test_camera_basis({0, 0, 5}, {0, 0, 0}, {0, 1, 0}, trace, "cam_standard");
    test_camera_basis({3, 4, 5}, {1, 0, -1}, {0, 1, 0}, trace, "cam_diagonal");
    test_camera_basis({-2, 1, 3}, {0, 0, 0}, {0, 0, 1}, trace, "cam_z_up");

    // Singularity tests: look direction parallel to up
    test_camera_basis({0, 5, 0}, {0, 0, 0}, {0, 1, 0}, trace, "cam_singularity_pos_y");
    test_camera_basis({0, -5, 0}, {0, 0, 0}, {0, 1, 0}, trace, "cam_singularity_neg_y");
    test_camera_basis({0, 0, 5}, {0, 0, 0}, {0, 0, 1}, trace, "cam_singularity_pos_z");
    test_camera_basis({0, 0, 0}, {0, 0, 0}, {0, 1, 0}, trace, "cam_degenerate_origin");

    // Save trace for Python inspection
    std::string trace_file = "trace_basis.json";
    bool saved = trace.save_json(trace_file);
    assert(saved);
    std::cout << "[PASS] All 1016 basis orthonormal and camera tests passed within 1e-12 tolerance!"
              << std::endl;
    std::cout << "[PASS] Saved trace records to " << trace_file << std::endl;

    return 0;
}
