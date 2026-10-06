#include "camera.hpp"
#include "hit_record.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "material.hpp"
#include "plane.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "stage_basis.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
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

Eigen::Vector3f ray_color(const mfad::Ray& r, const mfad::Hittable& scene, int depth) {
    if (depth <= 0) {
        return Eigen::Vector3f::Zero();
    }

    mfad::HitRecord rec;
    if (scene.hit(r, 0.001f, 1000.0f, rec)) {
        mfad::ScatterRecord srec;
        if (rec.material != nullptr && rec.material->scatter(r, rec, srec)) {
            return srec.attenuation.cwiseProduct(ray_color(srec.scattered, scene, depth - 1));
        }

        // Direct lighting fallback if no material attached
        Eigen::Vector3f light_dir = Eigen::Vector3f(-0.5f, 1.0f, 0.4f).normalized();
        float n_dot_l = std::max(0.0f, rec.normal.dot(light_dir));
        return (0.25f + 0.75f * n_dot_l) * rec.color;
    }

    // Sky background gradient
    float t = 0.5f * (r.direction.y() + 1.0f);
    return (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) + t * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer (Materials)    " << std::endl;
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
    const int samples_per_pixel = 16;
    const int max_depth = 8;

    mfad::ImageBuffer image(width, height);
    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.4f, 2.2f),   // Eye
                        Eigen::Vector3f(0.0f, 0.0f, -1.0f),  // Look-at
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f),   // Up
                        45.0f,                               // Vertical FOV
                        static_cast<float>(width) / static_cast<float>(height));

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f pixel_col = Eigen::Vector3f::Zero();
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u =
                    (static_cast<float>(x) + mfad::random_float()) / static_cast<float>(width - 1);
                float v = (static_cast<float>(height - 1 - y) + mfad::random_float()) /
                          static_cast<float>(height - 1);

                mfad::Ray ray = camera.generate_ray(u, v);
                pixel_col += ray_color(ray, scene, max_depth);
            }
            pixel_col /= static_cast<float>(samples_per_pixel);
            image.set_pixel(x, y, pixel_col);
        }
    }

    const std::string scene_output = "spheres_on_plane.png";
    if (image.write_png(scene_output)) {
        std::cout << "[SUCCESS] Rendered glass, mirror, and diffuse spheres on a plane to "
                  << scene_output << " (" << width << "x" << height << ", " << samples_per_pixel
                  << " spp)" << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << scene_output << std::endl;
        return 1;
    }

    return 0;
}
