#pragma once

#include "camera.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "material.hpp"
#include "ray.hpp"

#include <Eigen/Dense>

namespace mfad {

struct PathTracerOptions {
    int max_bounces = 32;
    int min_rr_bounces = 3;
    float rr_survival_clamp_min = 0.05f;
    float rr_survival_clamp_max = 0.95f;
    Eigen::Vector3f background_color = Eigen::Vector3f::Zero();
    bool use_sky_gradient = false;
};

/**
 * @brief Monte Carlo Path Tracer with iterative bouncing and Russian roulette termination.
 * Integrates the rendering equation using Duff et al. (2017) stage_basis frames.
 */
class PathTracer {
public:
    explicit PathTracer(const PathTracerOptions& options = PathTracerOptions())
        : options_(options) {}

    /**
     * @brief Computes radiance along a ray using iterative path tracing with Russian roulette.
     */
    Eigen::Vector3f trace_ray(const Ray& r, const Hittable& scene) const;

    /**
     * @brief Renders the entire image buffer with multi-threaded OpenMP execution.
     */
    void render(const Camera& camera, const Hittable& scene, ImageBuffer& buffer,
                int samples_per_pixel) const;

    const PathTracerOptions& options() const { return options_; }
    void set_options(const PathTracerOptions& options) { options_ = options; }

private:
    PathTracerOptions options_;
};

}  // namespace mfad
