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
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

void render_cornell_box(const std::string& out_path, const std::string& root_dir,
                        int samples_per_pixel = 128) {
    std::cout << "\n[TEST 2] Cornell Box Path Tracing Validation (Issue #26)..." << std::endl;
    std::cout << "[INFO] Target samples per pixel: " << samples_per_pixel << std::endl;
    std::string cb_scene_file = root_dir + "/scenes/cornell_box.json";
    if (!std::filesystem::exists(cb_scene_file)) {
        cb_scene_file = "scenes/cornell_box.json";
    }

    const int width = 640;
    const int height = 360;

    std::cout << "[INFO] Loading Cornell Box scene from " << cb_scene_file << std::endl;
    mfad::Scene cb_scene =
        mfad::SceneLoader::load_from_json(cb_scene_file, static_cast<double>(width) / height);
    cb_scene.print_summary();
    cb_scene.hittables.build_bvh();

    mfad::PathTracerOptions opts;
    opts.max_bounces = 16;
    opts.min_rr_bounces = 3;
    opts.background_color = Eigen::Vector3f::Zero();
    opts.use_sky_gradient = false;
    opts.sample_lights = true;
    for (auto& quad : cb_scene.area_lights) {
        opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
    }
    opts.point_lights = cb_scene.point_lights;

    mfad::PathTracer tracer(opts);
    mfad::ImageBuffer image(width, height);
    auto t_start = std::chrono::high_resolution_clock::now();
    tracer.render(*cb_scene.camera_data.camera, cb_scene.hittables, image, samples_per_pixel);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "[INFO] Cornell Box render complete in " << std::fixed << std::setprecision(2)
              << elapsed_sec << " s (" << (elapsed_sec / 60.0) << " min)" << std::endl;

    if (image.write_png(out_path)) {
        std::cout << "[SUCCESS] Saved Cornell Box to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
    }
}

void denoise_and_save(const mfad::ImageBuffer& pt_image, const std::string& denoised_out,
                      int svd_rank = 120) {
    int width = pt_image.width();
    int height = pt_image.height();
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
    }
}

void render_raw_frame(const std::string& out_path, const mfad::Scene& scene, int width,
                      int height) {
    std::cout << "\n[INFO] Stage 0: Rendering Raw Canvas (pre-pathtracing, unshadowed materials)..."
              << std::endl;
    mfad::ImageBuffer raw_image(width, height);
    mfad::DirectLightingOptions dl_opts;
    dl_opts.use_sky_gradient = false;
    dl_opts.background_color = Eigen::Vector3f(0.015f, 0.015f, 0.02f);
    dl_opts.ambient_color = Eigen::Vector3f(0.02f, 0.02f, 0.025f);
    dl_opts.enable_shadows =
        false;  // Zero shadows: authentic glass refraction & mirror reflection without occlusions

    mfad::render_direct_lighting(*scene.camera_data.camera, scene, raw_image, dl_opts, 32);

    if (raw_image.write_png(out_path)) {
        std::cout << "[SUCCESS] Saved Raw Canvas to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
    }
}

void run_quadruple_benchmark(const std::string& root_dir, const std::string& demo_scene_file,
                             int initial_spp = 32, double max_runtime_minutes = 20.0,
                             int max_spp = 0) {
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Quadrupling SPP Quality & Timing Benchmark     " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "[INFO] Loading unified scene from " << demo_scene_file << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(demo_scene_file);
    scene.print_summary();

    std::cout << "[INFO] Building BVH acceleration structure..." << std::endl;
    scene.hittables.build_bvh();

    const int width = 640;
    const int height = 360;

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

    double max_runtime_seconds = max_runtime_minutes * 60.0;
    int current_spp = initial_spp;
    int step_index = 1;

    struct BenchRecord {
        int spp;
        double seconds;
        std::string filename;
    };
    std::vector<BenchRecord> benchmark_log;

    while (true) {
        std::cout << "\n-------------------------------------------------" << std::endl;
        std::cout << "  [STEP " << step_index << "] Rendering " << current_spp << " spp (" << width
                  << "x" << height << ")..." << std::endl;
        std::cout << "-------------------------------------------------" << std::endl;

        mfad::ImageBuffer pt_image(width, height);

        auto t_start = std::chrono::high_resolution_clock::now();
        tracer.render(*scene.camera_data.camera, scene.hittables, pt_image, current_spp);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        double elapsed_min = elapsed_sec / 60.0;

        std::string out_filename = "pathtraced_" + std::to_string(current_spp) + "spp.png";
        std::string out_path = root_dir + "/" + out_filename;

        if (pt_image.write_png(out_path)) {
            std::cout << "[SUCCESS] Saved " << current_spp << " spp render to " << out_path
                      << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to save " << out_path << std::endl;
        }

        benchmark_log.push_back({current_spp, elapsed_sec, out_filename});

        std::cout << "[BENCHMARK RESULT] Step " << step_index << " (" << current_spp
                  << " spp): " << std::fixed << std::setprecision(2) << elapsed_sec << " s ("
                  << std::setprecision(2) << elapsed_min << " min)" << std::endl;

        // Save JSON benchmark record after each step
        std::ofstream json_out(root_dir + "/spp_benchmark_results.json");
        if (json_out.is_open()) {
            json_out << "[\n";
            for (size_t i = 0; i < benchmark_log.size(); ++i) {
                json_out << "  {\"spp\": " << benchmark_log[i].spp
                         << ", \"seconds\": " << std::fixed << std::setprecision(2)
                         << benchmark_log[i].seconds << ", \"minutes\": " << std::setprecision(2)
                         << (benchmark_log[i].seconds / 60.0) << ", \"file\": \""
                         << benchmark_log[i].filename << "\"}";
                if (i + 1 < benchmark_log.size())
                    json_out << ",";
                json_out << "\n";
            }
            json_out << "]\n";
            json_out.close();
        }

        if (max_spp > 0 && current_spp >= max_spp) {
            std::cout << "[STOPPING] Reached requested max SPP limit: " << max_spp << std::endl;
            break;
        }

        if (elapsed_sec >= max_runtime_seconds) {
            std::cout << "\n[STOPPING] Tracing time (" << elapsed_min
                      << " min) reached/exceeded target limit (" << max_runtime_minutes << " min)."
                      << std::endl;
            break;
        }

        double projected_next_sec = elapsed_sec * 4.0;
        std::cout << "[PROJECTION] Next step (" << (current_spp * 4) << " spp) estimated time: ~"
                  << std::setprecision(2) << (projected_next_sec / 60.0) << " min" << std::endl;

        current_spp *= 4;
        ++step_index;
    }

    std::cout << "\n=================================================" << std::endl;
    std::cout << "  BENCHMARK SUMMARY TABLE                        " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Step | SPP     | Tracing Time (s) | Tracing Time (min) | Relative Factor"
              << std::endl;
    std::cout << "-----------------------------------------------------------------------"
              << std::endl;
    for (size_t i = 0; i < benchmark_log.size(); ++i) {
        double rel = (i == 0) ? 1.0 : (benchmark_log[i].seconds / benchmark_log[i - 1].seconds);
        std::cout << (i + 1) << "    | " << std::setw(7) << benchmark_log[i].spp << " | "
                  << std::setw(16) << std::fixed << std::setprecision(2) << benchmark_log[i].seconds
                  << " s | " << std::setw(16) << (benchmark_log[i].seconds / 60.0) << " min | "
                  << std::setprecision(2) << rel << "x" << std::endl;
    }
    std::cout << "=================================================\n" << std::endl;
}

int main(int argc, char** argv) {
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

    // Check for command line arguments
    bool run_benchmark = false;
    bool cornell_only = false;
    bool demo_only = false;
    bool denoise_only = false;
    bool force_pt = false;
    int initial_spp = 32;
    int max_spp = 0;
    double max_runtime_min = 20.0;
    bool is_ci = (std::getenv("CI") != nullptr);
    int cornell_spp = is_ci ? 128 : 128;
    int pt_spp = is_ci ? 128 : 128;
    int direct_spp = 32;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark-spp" || arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--start-spp" && i + 1 < argc) {
            initial_spp = std::stoi(argv[++i]);
            run_benchmark = true;
        } else if (arg == "--max-spp" && i + 1 < argc) {
            max_spp = std::stoi(argv[++i]);
            run_benchmark = true;
        } else if (arg == "--max-time-min" && i + 1 < argc) {
            max_runtime_min = std::stod(argv[++i]);
            run_benchmark = true;
        } else if (arg == "--cornell-only") {
            cornell_only = true;
        } else if (arg == "--demo-only") {
            demo_only = true;
        } else if (arg == "--denoise-only") {
            denoise_only = true;
        } else if (arg == "--force" || arg == "--force-pt") {
            force_pt = true;
        } else if (arg == "--spp" && i + 1 < argc) {
            int val = std::stoi(argv[++i]);
            cornell_spp = val;
            pt_spp = val;
        } else if (arg == "--cornell-spp" && i + 1 < argc) {
            cornell_spp = std::stoi(argv[++i]);
        } else if ((arg == "--pt-spp" || arg == "--demo-spp") && i + 1 < argc) {
            pt_spp = std::stoi(argv[++i]);
        } else if (arg == "--direct-spp" && i + 1 < argc) {
            direct_spp = std::stoi(argv[++i]);
        }
    }

    if (run_benchmark) {
        run_quadruple_benchmark(root_dir, demo_scene_file, initial_spp, max_runtime_min, max_spp);
        return 0;
    }

    const std::string denoised_out = root_dir + "/final_demo_denoised.png";
    const std::string pathtraced_out = root_dir + "/final_demo_pathtraced.png";

    if (denoise_only) {
        std::string pt_file = pathtraced_out;
        if (!std::filesystem::exists(pt_file) &&
            std::filesystem::exists(root_dir + "/pathtraced_32768spp.png")) {
            pt_file = root_dir + "/pathtraced_32768spp.png";
        }
        mfad::ImageBuffer pt_buf(640, 360);
        if (pt_buf.read_png(pt_file)) {
            denoise_and_save(pt_buf, denoised_out, 120);
        } else {
            std::cerr << "[ERROR] Could not load " << pt_file << " for denoising" << std::endl;
            return 1;
        }
        return 0;
    }

    if (cornell_only) {
        const std::string cornell_out = root_dir + "/cornell_box.png";
        render_cornell_box(cornell_out, root_dir, cornell_spp);
        return 0;
    }

    if (!demo_only) {
        // 1. Initial Camera Ray / Sky Gradient verification test (Issue #19)
        const std::string gradient_out = root_dir + "/gradient.png";
        render_initial_gradient(gradient_out);

        // 2. Cornell Box Path Tracing benchmark test (Issue #26)
        const std::string cornell_out = root_dir + "/cornell_box.png";
        render_cornell_box(cornell_out, root_dir, cornell_spp);
    }

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

    const std::string raw_out = root_dir + "/final_demo_raw.png";
    const std::string direct_out = root_dir + "/final_demo_direct.png";

    // -------------------------------------------------------------
    // Stage 0: Raw Canvas Frame (pre-pathtracing, unshadowed geometry)
    // -------------------------------------------------------------
    render_raw_frame(raw_out, scene, width, height);

    // -------------------------------------------------------------
    // Stage 1: Direct Lighting (Whitted reflection/refraction + shadow rays)
    // -------------------------------------------------------------
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
    mfad::ImageBuffer pt_image(width, height);
    std::string preconverged_32k = root_dir + "/pathtraced_32768spp.png";

    if (pt_spp == 32768 && !force_pt && std::filesystem::exists(preconverged_32k)) {
        std::cout << "\n[INFO] Stage 2: Found pre-converged 32768 SPP render (" << preconverged_32k
                  << "). Promoting to " << pathtraced_out << "..." << std::endl;
        std::filesystem::copy_file(preconverged_32k, pathtraced_out,
                                   std::filesystem::copy_options::overwrite_existing);
        if (!pt_image.read_png(pathtraced_out)) {
            std::cerr << "[ERROR] Failed to load promoted " << pathtraced_out << std::endl;
            return 1;
        }
        std::cout << "[SUCCESS] Successfully promoted pre-converged 32768 SPP render to "
                  << pathtraced_out << std::endl;
    } else {
        std::cout << "\n[INFO] Stage 2: Rendering Path Traced Global Illumination (" << width << "x"
                  << height << ", " << pt_spp << " spp, max 16 bounces)..." << std::endl;
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
    }

    // -------------------------------------------------------------
    // Stage 3: SVD Low-Rank Denoising (Matrix truncated SVD)
    // -------------------------------------------------------------
    denoise_and_save(pt_image, denoised_out, 120);

    std::cout << "\n[ALL COMPLETE] Entire rendering process finished successfully!" << std::endl;
    return 0;
}
