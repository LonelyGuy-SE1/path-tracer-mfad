#include "camera.hpp"
#include "hit_record.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "plane.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "stage_basis.hpp"

#include <Eigen/Dense>
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

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(width - 1);
            float v = static_cast<float>(y) / static_cast<float>(height - 1);
            mfad::Ray ray = camera.generate_ray(u, v);
            Eigen::Vector3f col = 0.5f * (ray.direction + Eigen::Vector3f::Ones());
            image.set_pixel(x, y, col);
        }
    }
    image.write_png("gradient.png");
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer (Issue 19 & 21)" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    // 1. Generate gradient test artifact for Issue 19
    render_gradient_artifact();
    std::cout << "[SUCCESS] Rendered Issue 19 gradient to gradient.png" << std::endl;

    // 2. Set up 3D scene with Spheres on a Plane for Issue 21
    mfad::HittableList scene;

    // Floor plane: point (0, -0.5, 0), normal (0, 1, 0), light grey
    scene.add(std::make_shared<mfad::Plane>(Eigen::Vector3f(0.0f, -0.5f, 0.0f),
                                            Eigen::Vector3f(0.0f, 1.0f, 0.0f),
                                            Eigen::Vector3f(0.75f, 0.75f, 0.75f)));

    // Spheres: center red, left blue, right green, front gold
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, 0.0f, -1.2f), 0.5f,
                                             Eigen::Vector3f(0.85f, 0.22f, 0.20f)));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(-1.15f, 0.0f, -1.0f), 0.5f,
                                             Eigen::Vector3f(0.20f, 0.45f, 0.85f)));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(1.15f, 0.0f, -1.0f), 0.5f,
                                             Eigen::Vector3f(0.22f, 0.80f, 0.35f)));
    scene.add(std::make_shared<mfad::Sphere>(Eigen::Vector3f(0.0f, -0.25f, -0.5f), 0.25f,
                                             Eigen::Vector3f(0.95f, 0.75f, 0.15f)));

    // Camera setup: positioned so horizon is at mid-screen and sky is clearly visible
    const int width = 640;
    const int height = 360;
    mfad::ImageBuffer image(width, height);
    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.4f, 2.2f),   // Eye
                        Eigen::Vector3f(0.0f, 0.0f, -1.0f),  // Look-at
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f),   // Up
                        45.0f,                               // Vertical FOV
                        static_cast<float>(width) / static_cast<float>(height));

    // Directional key light for flat/diffuse shading
    const Eigen::Vector3f light_dir = Eigen::Vector3f(-0.5f, 1.0f, 0.4f).normalized();

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(width - 1);
            // Flip y so (0,0) is bottom-left in image coordinate space
            float v = static_cast<float>(height - 1 - y) / static_cast<float>(height - 1);

            mfad::Ray ray = camera.generate_ray(u, v);
            mfad::HitRecord rec;
            Eigen::Vector3f pixel_col;

            if (scene.hit(ray, 0.001f, 1000.0f, rec)) {
                // Base color: checkerboard grid for floor plane, solid color for spheres
                Eigen::Vector3f base_color = rec.color;
                if (std::abs(rec.normal.y() - 1.0f) < 1e-3f) {
                    float scale = 2.0f;
                    int cx = static_cast<int>(std::floor(rec.point.x() * scale));
                    int cz = static_cast<int>(std::floor(rec.point.z() * scale));
                    bool check = ((cx + cz) % 2 + 2) % 2 == 0;
                    base_color = check ? Eigen::Vector3f(0.85f, 0.85f, 0.88f)
                                       : Eigen::Vector3f(0.40f, 0.40f, 0.45f);
                }

                // Shading: ambient + diffuse Lambertian (n . l)
                float n_dot_l = std::max(0.0f, rec.normal.dot(light_dir));
                float intensity = 0.25f + 0.75f * n_dot_l;
                pixel_col = intensity * base_color;
            } else {
                // Background sky gradient from white (horizon) to deep sky blue
                float t = 0.5f * (ray.direction.y() + 1.0f);
                pixel_col = (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) +
                            t * Eigen::Vector3f(0.35f, 0.65f, 1.0f);
            }

            image.set_pixel(x, y, pixel_col);
        }
    }

    const std::string scene_output = "spheres_on_plane.png";
    if (image.write_png(scene_output)) {
        std::cout << "[SUCCESS] Rendered flat shaded spheres on a plane to " << scene_output << " ("
                  << width << "x" << height << ")" << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to save " << scene_output << std::endl;
        return 1;
    }

    return 0;
}
