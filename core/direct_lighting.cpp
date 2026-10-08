#include "direct_lighting.hpp"

#include "material.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace mfad {

Eigen::Vector3f
compute_direct_lighting(const Eigen::Vector3f& surface_point, const Eigen::Vector3f& normal,
                        const Eigen::Vector3f& albedo, const std::vector<PointLight>& lights,
                        const Hittable& scene, const DirectLightingOptions& options) {
    // Ambient illumination: L_ambient = albedo * ambient_color
    Eigen::Vector3f L = albedo.cwiseProduct(options.ambient_color);

    for (const auto& light : lights) {
        // Step 1: Use stage_shadow_direction to compute light direction, distance, and acne-free
        // offset origin
        Eigen::Vector3d pt_d = surface_point.cast<double>();
        Eigen::Vector3d light_pos_d = light.position.cast<double>();
        auto shadow_stage =
            stage_shadow_direction(pt_d, light_pos_d, options.shadow_epsilon, options.trace);
        const ShadowRayResult& s_res = shadow_stage.value;

        // Step 2: Cast shadow ray to test occlusion
        Ray shadow_ray(s_res.shadow_origin.cast<float>(), s_res.direction.cast<float>());
        float t_max = static_cast<float>(s_res.distance - options.shadow_epsilon);

        HitRecord occluder_rec;
        bool in_shadow =
            scene.hit(shadow_ray, static_cast<float>(options.shadow_epsilon), t_max, occluder_rec);

        if (!in_shadow) {
            // Step 3: Compute Lambertian diffuse cosine factor using stage_diffuse_term
            Eigen::Vector3d normal_d = normal.cast<double>();
            auto diffuse_stage = stage_diffuse_term(normal_d, s_res.direction, options.trace);
            float cos_theta = static_cast<float>(diffuse_stage.value);

            if (cos_theta > 0.0f) {
                // Step 4: Inverse-square distance attenuation (if enabled)
                float atten = 1.0f;
                if (options.use_distance_attenuation) {
                    float dist = static_cast<float>(s_res.distance);
                    float d_clamped = std::max(dist, options.min_distance_clamp);
                    atten = 1.0f / (d_clamped * d_clamped);
                }

                // Step 5: Direct diffuse reflection: L_direct = albedo * intensity * cos_theta *
                // atten
                L += (cos_theta * atten) * albedo.cwiseProduct(light.intensity);
            }
        }
    }

    return L;
}

void render_direct_lighting(const Camera& camera, const Scene& scene, ImageBuffer& buffer,
                            const DirectLightingOptions& options, int samples_per_pixel) {
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
                HitRecord rec;

                if (scene.hittables.hit(ray, 0.001f, 1e8f, rec)) {
                    // Extract material albedo
                    Eigen::Vector3f albedo = rec.color;
                    if (rec.material != nullptr) {
                        const auto* lambert = dynamic_cast<const Lambertian*>(rec.material);
                        if (lambert != nullptr) {
                            albedo = lambert->albedo();
                        } else {
                            ScatterRecord srec;
                            if (rec.material->scatter(ray, rec, srec)) {
                                albedo = srec.attenuation;
                            }
                        }
                    }

                    pixel_col +=
                        compute_direct_lighting(rec.point, rec.normal, albedo, scene.point_lights,
                                                scene.hittables, options);
                } else {
                    // Sky gradient for missed background rays
                    float t = 0.5f * (ray.direction.y() + 1.0f);
                    Eigen::Vector3f sky = (1.0f - t) * Eigen::Vector3f(1.0f, 1.0f, 1.0f) +
                                          t * Eigen::Vector3f(0.5f, 0.7f, 1.0f);
                    pixel_col += 0.2f * sky;
                }
            }

            pixel_col /= static_cast<float>(samples_per_pixel);
            buffer.set_pixel(x, y, pixel_col);
        }
    }
}

}  // namespace mfad
