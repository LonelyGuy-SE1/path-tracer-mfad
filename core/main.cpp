#include "camera.hpp"
#include "direct_lighting.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "material.hpp"
#include "path_tracer.hpp"
#include "quad.hpp"
#include "ray.hpp"
#include "scene_loader.hpp"
#include "stage_svd_denoise.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer                " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    // Determine paths relative to repo root
    std::string root_dir = ".";
    std::string demo_scene_file = "scenes/final_demo.json";
    if (!std::filesystem::exists(demo_scene_file) &&
        std::filesystem::exists("../" + demo_scene_file)) {
        root_dir = "..";
        demo_scene_file = "../" + demo_scene_file;
    }

    std::cout << "[INFO] Loading scene from " << demo_scene_file << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(demo_scene_file);
    scene.print_summary();

    // Construct BVH acceleration structure
    std::cout << "[INFO] Building BVH acceleration structure..." << std::endl;
    scene.hittables.build_bvh();

    const int width = 640;
    const int height = 360;

    // Output filenames in root directory
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

    std::cout << "\n[ALL COMPLETE] Rendering pipeline finished successfully!" << std::endl;
    return 0;
}
