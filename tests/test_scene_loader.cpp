#include "core/camera.hpp"
#include "core/scene_loader.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#define TEST_ASSERT(cond)                                                                          \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "Assertion failed at line " << __LINE__ << ": " #cond << std::endl;       \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#include <filesystem>

std::string find_scene_path(const std::string& rel) {
    if (std::filesystem::exists(rel)) {
        return rel;
    }
    if (std::filesystem::exists("../" + rel)) {
        return "../" + rel;
    }
    return rel;
}

void test_load_prep_demo_scene() {
    std::string path = find_scene_path("scenes/prep_demo.json");
    std::cout << "[TEST] Loading " << path << "..." << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(path, 16.0 / 9.0);

    // 1. Check Camera
    TEST_ASSERT(std::abs(scene.camera_data.eye.x() - 0.0) < 1e-6);
    TEST_ASSERT(std::abs(scene.camera_data.eye.y() - 0.4) < 1e-6);
    TEST_ASSERT(std::abs(scene.camera_data.eye.z() - 2.2) < 1e-6);
    TEST_ASSERT(std::abs(scene.camera_data.vfov - 45.0) < 1e-6);
    TEST_ASSERT(scene.camera_data.camera != nullptr);

    // 2. Check Materials
    TEST_ASSERT(scene.materials.size() == 2);
    TEST_ASSERT(scene.materials.count("red") == 1);
    TEST_ASSERT(scene.materials["red"].type == "diffuse");
    TEST_ASSERT(std::abs(scene.materials["red"].albedo.x() - 0.85) < 1e-5);
    TEST_ASSERT(scene.materials.count("glass") == 1);
    TEST_ASSERT(scene.materials["glass"].type == "glass");
    TEST_ASSERT(std::abs(scene.materials["glass"].ior - 1.5) < 1e-5);

    // 3. Check Meshes
    TEST_ASSERT(scene.meshes.size() == 1);
    const auto& mesh = scene.meshes[0];
    TEST_ASSERT(mesh.name == "cube");
    TEST_ASSERT(mesh.vertices.rows() == 3);
    TEST_ASSERT(mesh.vertices.cols() >= 8);
    TEST_ASSERT(mesh.faces.rows() >= 12);  // At least 12 triangles for a cube
    TEST_ASSERT(!mesh.triangles.empty());
    TEST_ASSERT(mesh.transform.rows() == 4 && mesh.transform.cols() == 4);

    // 4. Check Quadrics
    TEST_ASSERT(scene.quadrics.size() == 2);
    const auto& q0 = scene.quadrics[0];  // egg
    TEST_ASSERT(q0.name == "egg");
    TEST_ASSERT(q0.type == "ellipsoid");
    TEST_ASSERT(std::abs(q0.center.x() - (-1.1)) < 1e-5);
    TEST_ASSERT(std::abs(q0.radii.x() - 0.6) < 1e-5);
    TEST_ASSERT(q0.quadric_matrix.rows() == 4 && q0.quadric_matrix.cols() == 4);

    const auto& q1 = scene.quadrics[1];  // ball
    TEST_ASSERT(q1.name == "ball");
    TEST_ASSERT(q1.type == "sphere");
    TEST_ASSERT(std::abs(q1.center.x() - 1.1) < 1e-5);
    TEST_ASSERT(std::abs(q1.radii.x() - 0.5) < 1e-5);

    // 5. Check Object and Primitive Counts (Issue #20 requirement)
    TEST_ASSERT(scene.object_count() == 3);  // 1 mesh + 2 quadrics
    TEST_ASSERT(scene.total_primitive_count() >= 14);

    // 6. Check summary output
    std::stringstream ss;
    scene.print_summary(ss);
    std::string summary = ss.str();
    TEST_ASSERT(summary.find("Total Object Count: 3") != std::string::npos);

    std::cout << "[PASS] scenes/prep_demo.json loaded and validated successfully!" << std::endl;
}

void test_load_final_demo_scene() {
    std::string path = find_scene_path("scenes/final_demo.json");
    std::cout << "[TEST] Loading " << path << "..." << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(path, 16.0 / 9.0);

    TEST_ASSERT(scene.object_count() == 3);
    TEST_ASSERT(scene.meshes.size() == 1);
    TEST_ASSERT(scene.quadrics.size() == 2);
    TEST_ASSERT(scene.meshes[0].triangles.size() ==
                12);  // Clean cube.obj has 6 quad faces = 12 tris

    // Test ray-geometry intersection against loaded scene hittables
    // Ray towards ball (sphere at [1.1, 0, -1])
    mfad::Ray ray_ball(Eigen::Vector3f(1.1f, 0.0f, 2.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    mfad::HitRecord rec_ball;
    bool hit_ball = scene.hittables.hit(ray_ball, 0.001f, 100.0f, rec_ball);
    TEST_ASSERT(hit_ball);
    TEST_ASSERT(std::abs(rec_ball.point.x() - 1.1f) < 1e-3f);
    TEST_ASSERT(std::abs(rec_ball.point.z() - (-0.5f)) < 1e-3f);  // center - radius = -1 - (-0.5)

    // Ray towards egg (ellipsoid at [-1.1, 0, -1])
    mfad::Ray ray_egg(Eigen::Vector3f(-1.1f, 0.0f, 2.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));
    mfad::HitRecord rec_egg;
    bool hit_egg = scene.hittables.hit(ray_egg, 0.001f, 100.0f, rec_egg);
    TEST_ASSERT(hit_egg);

    // Miss ray
    mfad::Ray ray_miss(Eigen::Vector3f(10.0f, 10.0f, 10.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    mfad::HitRecord rec_miss;
    TEST_ASSERT(!scene.hittables.hit(ray_miss, 0.001f, 100.0f, rec_miss));

    std::cout << "[PASS] scenes/final_demo.json loaded and ray intersection verified!" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Running Scene Loader Tests (Issue #20)  " << std::endl;
    std::cout << "==========================================" << std::endl;

    test_load_prep_demo_scene();
    test_load_final_demo_scene();

    std::cout << "\n[PASS] All Scene Loader tests passed successfully!" << std::endl;
    return 0;
}
