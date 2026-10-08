#include "path_tracer.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace mfad {

Eigen::Vector3f PathTracer::trace_ray(const Ray& r, const Hittable& scene) const {
    Eigen::Vector3f L = Eigen::Vector3f::Zero();
    Eigen::Vector3f throughput = Eigen::Vector3f::Ones();
    Ray cur_ray = r;

    for (int bounce = 0; bounce < options_.max_bounces; ++bounce) {
        HitRecord rec;
        if (!scene.hit(cur_ray, 0.001f, 1e8f, rec)) {
            if (options_.use_sky_gradient) {
                float t = 0.5f * (cur_ray.direction.y() + 1.0f);
                Eigen::Vector3f sky = (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) +
                                      t * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
                L += throughput.cwiseProduct(sky);
            } else {
                L += throughput.cwiseProduct(options_.background_color);
            }
            break;
        }

        // Add emission from surface
        if (rec.material != nullptr) {
            L += throughput.cwiseProduct(rec.material->emitted(cur_ray, rec));
        }

        // Scatter ray
        ScatterRecord srec;
        if (rec.material == nullptr || !rec.material->scatter(cur_ray, rec, srec)) {
            break;
        }

        throughput = throughput.cwiseProduct(srec.attenuation);
        cur_ray = srec.scattered;

        // Russian Roulette termination (unbiased)
        if (bounce >= options_.min_rr_bounces) {
            // Photometric luminance: Y = 0.2126 R + 0.7152 G + 0.0722 B
            float p_survive =
                0.2126f * throughput.x() + 0.7152f * throughput.y() + 0.0722f * throughput.z();
            p_survive = std::clamp(p_survive, options_.rr_survival_clamp_min,
                                   options_.rr_survival_clamp_max);

            if (random_float() > p_survive) {
                break;  // Path terminated
            }
            throughput /= p_survive;  // Boost surviving throughput to maintain unbiased estimator
        }
    }

    return L;
}

void PathTracer::render(const Camera& camera, const Hittable& scene, ImageBuffer& buffer,
                        int samples_per_pixel) const {
    const int width = buffer.width();
    const int height = buffer.height();

#pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f pixel_col = Eigen::Vector3f::Zero();
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (static_cast<float>(x) + random_float()) / static_cast<float>(width - 1);
                // Scanline flip: bottom-left is (0, 0)
                float v = (static_cast<float>(height - 1 - y) + random_float()) /
                          static_cast<float>(height - 1);

                Ray ray = camera.generate_ray(u, v);
                pixel_col += trace_ray(ray, scene);
            }
            pixel_col /= static_cast<float>(samples_per_pixel);
            buffer.set_pixel(x, y, pixel_col);
        }
    }
}

}  // namespace mfad
