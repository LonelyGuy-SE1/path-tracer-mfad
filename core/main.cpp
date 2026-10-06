#include "camera.hpp"
#include "image_buffer.hpp"
#include "ray.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Path Tracer (MFAD) initialized." << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads available: " << omp_get_max_threads() << std::endl;

    const int width = 400;
    const int height = 225;
    mfad::ImageBuffer image(width, height);

    mfad::Camera camera(Eigen::Vector3f(0.0f, 0.0f, 2.0f),  // Eye position
                        Eigen::Vector3f(0.0f, 0.0f, 0.0f),  // Look-at point
                        Eigen::Vector3f(0.0f, 1.0f, 0.0f),  // Up vector
                        60.0f,                              // Vertical FOV in degrees
                        static_cast<float>(width) / static_cast<float>(height));

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(width - 1);
            float v = static_cast<float>(y) / static_cast<float>(height - 1);

            mfad::Ray ray = camera.generate_ray(u, v);

            // Direction color gradient: normalized direction mapped to [0, 1] RGB
            Eigen::Vector3f col = 0.5f * (ray.direction + Eigen::Vector3f::Ones());
            image.set_pixel(x, y, col);
        }
    }

    const std::string out_path = "gradient.png";
    if (image.write_png(out_path)) {
        std::cout << "[SUCCESS] Rendered flat colour gradient to " << out_path << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to write " << out_path << std::endl;
        return 1;
    }

    return 0;
}
