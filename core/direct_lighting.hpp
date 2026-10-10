#pragma once

#include "camera.hpp"
#include "image_buffer.hpp"
#include "light.hpp"
#include "scene.hpp"
#include "stages/stage_projection.hpp"
#include "trace/trace.hpp"

#include <Eigen/Dense>
#include <vector>

namespace mfad {

struct DirectLightingOptions {
    Eigen::Vector3f ambient_color{0.02f, 0.02f, 0.02f};
    bool use_distance_attenuation{true};
    float min_distance_clamp{0.3f};
    double shadow_epsilon{1e-4};
    Trace* trace{nullptr};
    bool use_sky_gradient{false};
    Eigen::Vector3f background_color{Eigen::Vector3f::Zero()};
};

/**
 * @brief Computes Lambertian diffuse direct lighting with point lights and shadow rays (Issue
 * #24). Uses stage_projection algorithms (stage_shadow_direction and stage_diffuse_cosine).
 *
 * @param surface_point 3D world position on surface.
 * @param normal Surface unit normal vector.
 * @param albedo Surface diffuse reflectance (color).
 * @param lights List of point lights in the scene.
 * @param scene Scene hittables for shadow ray occlusion testing.
 * @param options Lighting options (ambient, attenuation, tracing).
 * @return Reflected radiance vector (RGB).
 */
Eigen::Vector3f
compute_direct_lighting(const Eigen::Vector3f& surface_point, const Eigen::Vector3f& normal,
                        const Eigen::Vector3f& albedo, const std::vector<PointLight>& lights,
                        const Hittable& scene,
                        const DirectLightingOptions& options = DirectLightingOptions());

/**
 * @brief Renders an entire scene using direct Lambertian lighting with ray-traced shadows.
 *
 * @param camera Scene camera.
 * @param scene Scene containing geometry, materials, and point lights.
 * @param buffer Output image buffer.
 * @param options Lighting options.
 * @param samples_per_pixel Anti-aliasing super-sampling count per pixel.
 */
void render_direct_lighting(const Camera& camera, const Scene& scene, ImageBuffer& buffer,
                            const DirectLightingOptions& options = DirectLightingOptions(),
                            int samples_per_pixel = 4);

}  // namespace mfad
