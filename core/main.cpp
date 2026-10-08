#include "camera.hpp"
#include "direct_lighting.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "material.hpp"
#include "path_tracer.hpp"
#include "plane.hpp"
#include "quad.hpp"
#include "ray.hpp"
#include "scene_loader.hpp"
#include "sphere.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <memory>
#include <omp.h>

void render_gradient_artifact() {
    const int width = 400;
    const int height = 225;
    mfad::ImageBuffer image(width, height);
    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.0f, 2.0f), Eigen::Vector3f(0.0f, 0.0f, 0.0f),
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f), 60.0f,
                        static_cast<float>(width) / static_cast<float>(height));

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(width - 1);
            // Flip y so bottom-left is (0, 0), matching the scene convention
            float v = static_cast<float>(height - 1 - y) / static_cast<float>(height - 1);
            mfad::Ray ray = camera.generate_ray(u, v);

            // Classic sky gradient: smooth transition from white (horizon) to blue (zenith)
            float t = 0.5f * (ray.direction.y() + 1.0f);
            Eigen::Vector3f col = (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) +
                                  t * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
            image.set_pixel(x, y, col);
        }
    }
    image.write_png("gradient.png");
}

void render_cornell_box() {
    const int width = 400;
    const int height = 400;
    const int samples_per_pixel = 256;

    mfad::HittableList scene;

    // Materials
    auto mat_red = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.65f, 0.05f, 0.05f));
    auto mat_green = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.12f, 0.45f, 0.15f));
    auto mat_white = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.73f, 0.73f, 0.73f));
    auto mat_light = std::make_shared<mfad::DiffuseLight>(Eigen::Vector3f(15.0f, 15.0f, 15.0f),
                                                          /*two_sided=*/true);
    auto mat_glass = std::make_shared<mfad::Dielectric>(1.5f);
    auto mat_mirror = std::make_shared<mfad::Metal>(Eigen::Vector3f(0.9f, 0.9f, 0.9f), 0.0f);

    // Floor (pointing up +y, from z = -2.5 to z = 1.5)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-1.0f, -1.0f, 1.5f), Eigen::Vector3f(2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, -4.0f), Eigen::Vector3f::Ones(), mat_white));

    // Ceiling (pointing down -y)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-1.0f, 1.0f, -2.5f), Eigen::Vector3f(2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 4.0f), Eigen::Vector3f::Ones(), mat_white));

    // Ceiling Light (pointing down -y)
    auto ceiling_light = std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-0.35f, 0.999f, -1.8f), Eigen::Vector3f(0.7f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 0.0f, 0.7f), Eigen::Vector3f::Ones(), mat_light);
    scene.add(ceiling_light);

    // Back wall (pointing forward +z)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-1.0f, -1.0f, -2.5f), Eigen::Vector3f(2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f::Ones(), mat_white));

    // Front wall behind camera (pointing backward -z)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(1.0f, -1.0f, 1.5f), Eigen::Vector3f(-2.0f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f::Ones(), mat_white));

    // Left wall (Red, pointing right +x)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-1.0f, -1.0f, 1.5f), Eigen::Vector3f(0.0f, 0.0f, -4.0f),
        Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f::Ones(), mat_red));

    // Right wall (Green, pointing left -x)
    scene.add(std::make_shared<mfad::Quad>(
        Eigen::Vector3f(1.0f, -1.0f, -2.5f), Eigen::Vector3f(0.0f, 0.0f, 4.0f),
        Eigen::Vector3f(0.0f, 2.0f, 0.0f), Eigen::Vector3f::Ones(), mat_green));

    // Spheres inside box: glass sphere on left, chrome mirror sphere on right
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(-0.45f, -0.6f, -1.4f), 0.4f,
                                             Eigen::Vector3f::Ones(), mat_glass));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.45f, -0.6f, -1.8f), 0.4f,
                                             Eigen::Vector3f::Ones(), mat_mirror));

    // Camera setup for Cornell box
    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.0f, 1.4f), Eigen::Vector3f(0.0f, 0.0f, -1.5f),
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f), 55.0f, 1.0f);

    mfad::PathTracerOptions opts;
    opts.max_bounces = 32;
    opts.min_rr_bounces = 3;
    opts.rr_survival_clamp_min = 0.05f;
    opts.rr_survival_clamp_max = 0.95f;
    opts.background_color = Eigen::Vector3f::Zero();
    opts.use_sky_gradient = false;
    opts.sample_lights = true;
    opts.area_lights.push_back(ceiling_light);

    mfad::PathTracer tracer(opts);
    mfad::ImageBuffer image(width, height);

    std::cout << "[INFO] Rendering Cornell Box (" << width << "x" << height << ", "
              << samples_per_pixel << " spp, max 32 bounces with Russian roulette)..." << std::endl;
    tracer.render(camera, scene, image, samples_per_pixel);

    const std::string out_path = "cornell_box.png";
    if (image.write_png(out_path)) {
        std::cout << "[SUCCESS] Rendered Cornell Box to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
    }
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer (Issue #26)    " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    // 1. Generate gradient test artifact for Issue 19
    render_gradient_artifact();
    std::cout << "[SUCCESS] Rendered Issue 19 gradient to gradient.png" << std::endl;

    // 2. Set up 3D scene with Diffuse, Glass, Mirror, and Checker floor
    mfad::HittableList scene;

    // Materials
    auto mat_floor = std::make_shared<mfad::CheckerMaterial>(
        Eigen::Vector3f(0.85f, 0.85f, 0.88f), Eigen::Vector3f(0.35f, 0.35f, 0.40f), 2.0f);
    auto mat_diffuse_red = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.85f, 0.22f, 0.20f));
    auto mat_glass = std::make_shared<mfad::Dielectric>(1.5f);  // Real glass (n = 1.5)
    auto mat_mirror = std::make_shared<mfad::Metal>(Eigen::Vector3f(0.85f, 0.88f, 0.90f),
                                                    0.02f);  // Chrome mirror
    auto mat_gold =
        std::make_shared<mfad::Metal>(Eigen::Vector3f(0.95f, 0.75f, 0.15f), 0.05f);  // Gold mirror

    // Floor plane
    scene.add(std::make_shared<mfad::Plane>(Eigen::Vector3f(0.0f, -0.5f, 0.0f),
                                            Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                                            Eigen::Vector3f::Ones(), mat_floor));

    // Spheres: center diffuse red, left glass, right chrome mirror, front gold mirror
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, -1.2f), 0.5f,
                                             Eigen::Vector3f::Ones(), mat_diffuse_red));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(-1.15f, 0.0f, -1.0f), 0.5f,
                                             Eigen::Vector3f::Ones(), mat_glass));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(1.15f, 0.0f, -1.0f), 0.5f,
                                             Eigen::Vector3f::Ones(), mat_mirror));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, -0.25f, -0.5f), 0.25f,
                                             Eigen::Vector3f::Ones(), mat_gold));

    // Camera setup
    const int width = 640;
    const int height = 360;
    const int samples_per_pixel = 128;

    mfad::ImageBuffer image(width, height);
    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.4f, 2.2f),   // Eye
                        Eigen::Vector3f(0.0f, 0.0f, -1.0f),  // Look-at
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f),   // Up
                        45.0f,                               // Vertical FOV
                        static_cast<float>(width) / static_cast<float>(height));

    // Singular studio key light source for dramatic lighting
    auto mat_studio_light = std::make_shared<mfad::DiffuseLight>(
        Eigen::Vector3f(35.0f, 35.0f, 35.0f), /*two_sided=*/true);
    auto studio_quad = std::make_shared<mfad::Quad>(
        Eigen::Vector3f(-0.5f, 4.0f, 0.2f), Eigen::Vector3f(2.5f, 0.0f, 0.0f),
        Eigen::Vector3f(0.0f, -0.4f, 2.0f), Eigen::Vector3f::Ones(), mat_studio_light);
    scene.add(studio_quad);

    mfad::PathTracerOptions opts_studio;
    opts_studio.max_bounces = 16;
    opts_studio.min_rr_bounces = 3;
    opts_studio.use_sky_gradient = false;
    opts_studio.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    opts_studio.sample_lights = true;
    opts_studio.area_lights.push_back(studio_quad);

    mfad::PathTracer tracer(opts_studio);
    std::cout << "[INFO] Rendering spheres scene with singular key light (" << width << "x"
              << height << ", " << samples_per_pixel << " spp)..." << std::endl;
    tracer.render(camera, scene, image, samples_per_pixel);

    const std::string scene_output = "spheres_on_plane.png";
    if (image.write_png(scene_output)) {
        std::cout << "[SUCCESS] Rendered glass, mirror, and diffuse spheres on a plane to "
                  << scene_output << " (" << width << "x" << height << ", " << samples_per_pixel
                  << " spp)" << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << scene_output << std::endl;
        return 1;
    }

    // 3. Render Cornell Box (Issue #26)
    render_cornell_box();

    // 4. Scene Loader & Direct Lighting Demo (Issue #20 & Issue #24)
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Scene Loader & Direct Lighting (Issue #20, #24) " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::string demo_scene_file = "scenes/final_demo.json";
    if (!std::filesystem::exists(demo_scene_file) &&
        std::filesystem::exists("../" + demo_scene_file)) {
        demo_scene_file = "../" + demo_scene_file;
    }
    mfad::Scene demo_scene = mfad::SceneLoader::load_from_json(demo_scene_file);
    demo_scene.print_summary();

    // Add floor plane and singular corner light for direct lighting verification
    auto demo_floor_mat = std::make_shared<mfad::CheckerMaterial>(
        Eigen::Vector3f(0.85f, 0.85f, 0.88f), Eigen::Vector3f(0.35f, 0.35f, 0.40f), 2.0f);
    demo_scene.hittables.add(std::make_shared<mfad::Plane>(
        Eigen::Vector3f(0.0f, -0.5f, 0.0f), Eigen::Vector3f(0.0f, 1.0f, 0.0f),
        Eigen::Vector3f::Ones(), demo_floor_mat));
    demo_scene.point_lights.clear();
    demo_scene.point_lights.emplace_back(Eigen::Vector3f(2.2f, 2.8f, 0.5f),
                                         Eigen::Vector3f(40.0f, 40.0f, 40.0f));

    const int demo_width = 640;
    const int demo_height = 360;
    mfad::ImageBuffer direct_image(demo_width, demo_height);
    mfad::DirectLightingOptions dl_opts;
    dl_opts.use_sky_gradient = false;
    dl_opts.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    dl_opts.ambient_color = Eigen::Vector3f(0.02f, 0.02f, 0.025f);

    std::cout << "[INFO] Rendering loaded scene with Direct Lighting (singular corner light)..."
              << std::endl;
    mfad::render_direct_lighting(*demo_scene.camera_data.camera, demo_scene, direct_image, dl_opts,
                                 16);
    const std::string direct_out = "final_demo_direct.png";
    if (direct_image.write_png(direct_out)) {
        std::cout << "[SUCCESS] Rendered final demo direct lighting to " << direct_out << std::endl;
    }

    // Add singular physical key area light in the upper corner for Monte Carlo Path Tracer
    auto mat_demo_light = std::make_shared<mfad::DiffuseLight>(Eigen::Vector3f(40.0f, 40.0f, 40.0f),
                                                               /*two_sided=*/true);
    auto demo_corner_light = std::make_shared<mfad::Quad>(
        Eigen::Vector3f(1.8f, 2.4f, 0.1f), Eigen::Vector3f(0.8f, 0.0f, 0.6f),
        Eigen::Vector3f(0.0f, 0.8f, -0.3f), Eigen::Vector3f::Ones(), mat_demo_light);
    demo_scene.hittables.add(demo_corner_light);

    // Render loaded demo scene with full Monte Carlo Path Tracer using Next Event Estimation
    const int pt_spp = 512;
    std::cout << "[INFO] Rendering loaded scene with PathTracer (singular corner light, NEE, "
              << pt_spp << " spp)..." << std::endl;
    mfad::ImageBuffer pt_image(demo_width, demo_height);
    mfad::PathTracerOptions pt_opts;
    pt_opts.use_sky_gradient = false;
    pt_opts.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    pt_opts.max_bounces = 16;
    pt_opts.min_rr_bounces = 3;
    pt_opts.sample_lights = true;
    pt_opts.area_lights.push_back(demo_corner_light);

    mfad::PathTracer demo_tracer(pt_opts);
    demo_tracer.render(*demo_scene.camera_data.camera, demo_scene.hittables, pt_image, pt_spp);
    const std::string pt_out = "final_demo_pathtraced.png";
    if (pt_image.write_png(pt_out)) {
        std::cout << "[SUCCESS] Rendered final demo path tracing to " << pt_out << std::endl;
    }

    return 0;
}
