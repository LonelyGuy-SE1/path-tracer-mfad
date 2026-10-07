#include "stage_projection.hpp"
#include "trace.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void check_normal_projection(const Eigen::Vector3d& v, const Eigen::Vector3d& n, mfad::Trace& trace,
                             int index) {
    auto res = mfad::stage_projection_normal(v, n, &trace, "normal_" + std::to_string(index));
    const Eigen::Vector3d nu = n.normalized();

    // P_normal is idempotent and symmetric, and splits v exactly.
    assert((res.P_normal * res.P_normal - res.P_normal).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.P_normal - res.P_normal.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.along_normal + res.in_tangent_plane - v).cwiseAbs().maxCoeff() < 1e-12);
    assert(std::abs(res.in_tangent_plane.dot(nu)) < 1e-12 * std::max(1.0, v.norm()));

    // Reflection keeps the length, flips the normal component, keeps the tangent component.
    assert(std::abs(res.reflected.norm() - v.norm()) < 1e-12 * std::max(1.0, v.norm()));
    assert(std::abs(res.reflected.dot(nu) + v.dot(nu)) < 1e-12 * std::max(1.0, v.norm()));
}

void check_subspace_projection(const Eigen::MatrixXd& A, const Eigen::Vector3d& v,
                               mfad::Trace& trace, const std::string& name) {
    auto res = mfad::stage_projection_subspace(A, v, &trace, name);

    assert((res.P * res.P - res.P).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.P - res.P.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    assert(std::abs(res.P.trace() - static_cast<double>(A.cols())) < 1e-12);
    assert((A.transpose() * res.perpendicular).cwiseAbs().maxCoeff() < 1e-12);
    assert((res.parallel + res.perpendicular - v).cwiseAbs().maxCoeff() < 1e-12);
}

}  // namespace

int main() {
    std::cout << "[TEST] Running stage_projection verification tests..." << std::endl;
    mfad::Trace trace;

    // Fixed cases: axis-aligned normals, a grazing ray, and an unnormalized normal.
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> fixed = {
        {{0, 0, -1}, {0, 0, 1}},        // head-on
        {{1, 0, 0}, {0, 0, 1}},         // grazing: no normal component
        {{1, -1, -1}, {0, 0, 1}},       // 45 degree-ish bounce off the floor
        {{0.3, -0.8, 0.5}, {0, 2, 0}},  // unnormalized normal
        {{-2.0, 3.0, 1.0}, {1, 1, 1}},  // oblique
    };
    int idx = 0;
    for (const auto& [v, n] : fixed) {
        check_normal_projection(v, n, trace, idx++);
    }

    // Random incoming directions and normals.
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < 500; ++i) {
        Eigen::Vector3d v(dist(rng), dist(rng), dist(rng));
        Eigen::Vector3d n(dist(rng), dist(rng), dist(rng));
        check_normal_projection(v, n, trace, idx++);
    }

    // Subspace projections: a line, a plane (the image plane [right, up]) and a random plane.
    Eigen::MatrixXd line(3, 1);
    line << 1.0, 2.0, -2.0;
    check_subspace_projection(line, Eigen::Vector3d(3.0, -1.0, 2.0), trace, "line");

    Eigen::MatrixXd image_plane(3, 2);
    image_plane << 1, 0, 0, 1, 0, 0;  // right = x, up = y
    check_subspace_projection(image_plane, Eigen::Vector3d(2.0, 3.0, -5.0), trace, "image_plane");

    for (int i = 0; i < 200; ++i) {
        Eigen::MatrixXd A(3, 2);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 2; ++c) {
                A(r, c) = dist(rng);
            }
        }
        Eigen::Vector3d v(dist(rng), dist(rng), dist(rng));
        check_subspace_projection(A, v, trace, "plane_" + std::to_string(i));
    }

    // Dependent columns must be rejected.
    bool threw = false;
    try {
        Eigen::MatrixXd dep(3, 2);
        dep << 1, 2, 0, 0, 0, 0;
        mfad::stage_projection_subspace(dep, Eigen::Vector3d(1, 1, 1));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    bool saved = trace.save_json("trace_projection.json");
    assert(saved);
    std::cout << "[PASS] All stage_projection tests passed within 1e-12 tolerance!" << std::endl;
    std::cout << "[PASS] Saved trace records to trace_projection.json" << std::endl;
    return 0;
}
