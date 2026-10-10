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

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer                " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    std::string demo_scene_file = "scenes/final_demo.json";
    if (!std::filesystem::exists(demo_scene_file) &&
        std::filesystem::exists("../" + demo_scene_file)) {
        demo_scene_file = "../" + demo_scene_file;
    }
    
    std::cout << "[INFO] Loading unified scene from " << demo_scene_file << std::endl;
    mfad::Scene scene = mfad::SceneLoader::load_from_json(demo_scene_file);
    scene.print_summary();
    scene.hittables.build_bvh();

    const int width = 640;
    const int height = 360;

    // We will render this EXACT SAME SCENE 5 times to show different rendering features.
    
    // Step 1: Direct Lighting (Low Quality, 1 spp)
    {
        mfad::ImageBuffer image(width, height);
        mfad::DirectLightingOptions opts;
        opts.use_sky_gradient = true;
        opts.disable_shadows = true;
        std::cout << "[INFO] Step 1: Rendering Direct Lighting (1 spp, No Shadows)..." << std::endl;
        mfad::render_direct_lighting(*scene.camera_data.camera, scene, image, opts, 1);
        image.write_png("step1_direct_1spp.png");
    }

    // Step 2: Direct Lighting (High Quality, 32 spp)
    {
        mfad::ImageBuffer image(width, height);
        mfad::DirectLightingOptions opts;
        opts.use_sky_gradient = true;
        std::cout << "[INFO] Step 2: Rendering Direct Lighting (32 spp)..." << std::endl;
        mfad::render_direct_lighting(*scene.camera_data.camera, scene, image, opts, 32);
        image.write_png("step2_direct_32spp.png");
    }

    // Step 3: Path Tracing (No bounces, i.e. max_bounces=1)
    {
        mfad::ImageBuffer image(width, height);
        mfad::PathTracerOptions opts;
        opts.use_sky_gradient = true;
        opts.max_bounces = 2;
        opts.sample_lights = true;
        for(auto& quad : scene.area_lights) opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
        opts.point_lights = scene.point_lights;
        
        mfad::PathTracer tracer(opts);
        std::cout << "[INFO] Step 3: Rendering Path Tracer (2 bounces, 16 spp)..." << std::endl;
        tracer.render(*scene.camera_data.camera, scene.hittables, image, 16);
        image.write_png("step3_pathtraced_1bounce.png");
    }

    // Step 4: Path Tracing (Global Illumination, 16 spp)
    {
        mfad::ImageBuffer image(width, height);
        mfad::PathTracerOptions opts;
        opts.use_sky_gradient = true;
        opts.max_bounces = 16;
        opts.sample_lights = true;
        for(auto& quad : scene.area_lights) opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
        opts.point_lights = scene.point_lights;

        mfad::PathTracer tracer(opts);
        std::cout << "[INFO] Step 4: Rendering Path Tracer (16 bounces, 16 spp)..." << std::endl;
        tracer.render(*scene.camera_data.camera, scene.hittables, image, 16);
        image.write_png("step4_pathtraced_16spp.png");
    }

    // Step 5: Path Tracing (Global Illumination, High Quality, 128 spp)
    {
        mfad::ImageBuffer image(width, height);
        mfad::PathTracerOptions opts;
        opts.use_sky_gradient = true;
        opts.max_bounces = 16;
        opts.sample_lights = true;
        for(auto& quad : scene.area_lights) opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
        opts.point_lights = scene.point_lights;

        mfad::PathTracer tracer(opts);
        std::cout << "[INFO] Step 5: Rendering Path Tracer (16 bounces, 128 spp)..." << std::endl;
        tracer.render(*scene.camera_data.camera, scene.hittables, image, 128);
        image.write_png("step5_pathtraced_128spp.png");
    }

    return 0;
}
