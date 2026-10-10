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
#include "stage_svd_denoise.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

void render_initial_gradient(const std::string& out_path) {
    std::cout << "\n[TEST 1] Initial Camera & Ray Verification (Issue #19)..." << std::endl;
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
            float v = static_cast<float>(height - 1 - y) / static_cast<float>(height - 1);
            mfad::Ray ray = camera.generate_ray(u, v);

            // Classic sky gradient: smooth transition from white (horizon) to blue (zenith)
            float t = 0.5f * (ray.direction.y() + 1.0f);
            Eigen::Vector3f col = (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) +
                                  t * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
            image.set_pixel(x, y, col);
        }
    }

    if (image.write_png(out_path)) {
        std::cout << "[SUCCESS] Saved Initial Camera Gradient to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
    }
}

void render_cornell_box(const std::string& out_path) {
    std::cout << "\n[TEST 2] Cornell Box Path Tracing Validation (Issue #26)..." << std::endl;
    const int width = 400;
    const int height = 400;
    const int samples_per_pixel = 128;

    mfad::HittableList scene;

    // Materials
    auto mat_red = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.65f, 0.05f, 0.05f));
    auto mat_green = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.12f, 0.45f, 0.15f));
    auto mat_white = std::make_shared<mfad::Lambertian>(Eigen::Vector3f(0.73f, 0.73f, 0.73f));
    auto mat_light = std::make_shared<mfad::DiffuseLight>(Eigen::Vector3f(15.0f, 15.0f, 15.0f),
                                                          /*two_sided=*/true);
    auto mat_glass = std::make_shared<mfad::Dielectric>(1.5f);
    auto mat_mirror = std::make_shared<mfad::Metal>(Eigen::Vector3f(0.9f, 0.9f, 0.9f), 0.0f);

    // Floor (pointing up +y)
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
    tracer.render(camera, scene, image, samples_per_pixel);

    if (image.write_png(out_path)) {
        std::cout << "[SUCCESS] Saved Cornell Box to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
    }
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer                " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    // Determine repo root directory so all outputs land in ONE consistent place
    std::string root_dir = ".";
    std::string demo_scene_file = "scenes/final_demo.json";
    if (!std::filesystem::exists(demo_scene_file) &&
        std::filesystem::exists("../" + demo_scene_file)) {
        root_dir = "..";
        demo_scene_file = "../" + demo_scene_file;
    }

    // 1. Initial Camera Ray / Sky Gradient verification test (Issue #19)
    const std::string gradient_out = root_dir + "/gradient.png";
    render_initial_gradient(gradient_out);

    // 2. Cornell Box Path Tracing benchmark test (Issue #26)
    const std::string cornell_out = root_dir + "/cornell_box.png";
    render_cornell_box(cornell_out);

    // 3. Unified Final Demo Scene Pipeline (Issue #7, #8, #18, #20, #24)
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Unified Scene Pipeline (scenes/final_demo.json)" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "[INFO] Loading unified scene from " << demo_scene_file << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(demo_scene_file);
    scene.print_summary();

    // Construct BVH acceleration structure
    std::cout << "[INFO] Building BVH acceleration structure..." << std::endl;
    scene.hittables.build_bvh();

    const int width = 640;
    const int height = 360;

    const std::string direct_out = root_dir + "/final_demo_direct.png";
    const std::string pathtraced_out = root_dir + "/final_demo_pathtraced.png";
    const std::string denoised_out = root_dir + "/final_demo_denoised.png";

    // -------------------------------------------------------------
    // Stage 1: Direct Lighting (Whitted reflection/refraction + shadow rays)
    // -------------------------------------------------------------
    const int direct_spp = 32;
    std::cout << "\n[INFO] Stage 1: Rendering Direct Lighting (" << width << "x" << height << ", "
              << direct_spp << " spp)..." << std::endl;
    mfad::ImageBuffer direct_image(width, height);
    mfad::DirectLightingOptions dl_opts;
    dl_opts.use_sky_gradient = false;
    dl_opts.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    dl_opts.ambient_color = Eigen::Vector3f(0.02f, 0.02f, 0.025f);

    mfad::render_direct_lighting(*scene.camera_data.camera, scene, direct_image, dl_opts,
                                 direct_spp);
    if (direct_image.write_png(direct_out)) {
        std::cout << "[SUCCESS] Saved Direct Lighting to " << direct_out << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << direct_out << std::endl;
        return 1;
    }

    // -------------------------------------------------------------
    // Stage 2: Monte Carlo Path Tracing (Global Illumination + NEE + RR)
    // -------------------------------------------------------------
    const int pt_spp = 128;
    std::cout << "\n[INFO] Stage 2: Rendering Path Traced Global Illumination (" << width << "x"
              << height << ", " << pt_spp << " spp, max 16 bounces)..." << std::endl;
    mfad::ImageBuffer pt_image(width, height);
    mfad::PathTracerOptions pt_opts;
    pt_opts.use_sky_gradient = false;
    pt_opts.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    pt_opts.max_bounces = 16;
    pt_opts.min_rr_bounces = 3;
    pt_opts.sample_lights = true;
    for (auto& quad : scene.area_lights) {
        pt_opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
    }
    pt_opts.point_lights = scene.point_lights;

    mfad::PathTracer tracer(pt_opts);
    tracer.render(*scene.camera_data.camera, scene.hittables, pt_image, pt_spp);
    if (pt_image.write_png(pathtraced_out)) {
        std::cout << "[SUCCESS] Saved Path Tracing to " << pathtraced_out << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << pathtraced_out << std::endl;
        return 1;
    }

    // -------------------------------------------------------------
    // Stage 3: SVD Low-Rank Denoising (Matrix truncated SVD)
    // -------------------------------------------------------------
    const int svd_rank = 120;
    std::cout << "\n[INFO] Stage 3: Applying SVD Low-Rank Denoising (Rank " << svd_rank << ")..."
              << std::endl;
    std::vector<Eigen::MatrixXd> channels(3, Eigen::MatrixXd(height, width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f c = pt_image.get_pixel(x, y);
            channels[0](y, x) = static_cast<double>(c.x());
            channels[1](y, x) = static_cast<double>(c.y());
            channels[2](y, x) = static_cast<double>(c.z());
        }
    }

    auto denoise_res = mfad::stage_svd_denoise_image(channels, svd_rank);
    mfad::ImageBuffer denoised_image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[0](y, x)));
            float g =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[1](y, x)));
            float b =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[2](y, x)));
            denoised_image.set_pixel(x, y, Eigen::Vector3f(r, g, b));
        }
    }

    if (denoised_image.write_png(denoised_out)) {
        std::cout << "[SUCCESS] Saved SVD Denoised image to " << denoised_out << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << denoised_out << std::endl;
        return 1;
    }

    std::cout << "\n[ALL COMPLETE] Entire rendering process finished successfully!" << std::endl;
    return 0;
}
