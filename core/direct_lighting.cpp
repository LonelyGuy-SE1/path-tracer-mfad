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

namespace {

Eigen::Vector3f shade_direct_ray(const Ray& ray, const Scene& scene,
                                 const DirectLightingOptions& options, int depth) {
    if (depth > 8) {
        return Eigen::Vector3f::Zero();
    }

    HitRecord rec;
    if (!scene.hittables.hit(ray, 0.001f, 1e8f, rec)) {
        // Direct lighting sky model: scaled to harmoniously match ambient/direct lighting levels
        const Eigen::Vector3f sky_zenith(0.12f, 0.18f, 0.28f);
        const Eigen::Vector3f sky_horizon(0.24f, 0.28f, 0.35f);
        const Eigen::Vector3f ground_nadir(0.08f, 0.08f, 0.10f);

        float dir_y = ray.direction.y();
        if (dir_y >= 0.0f) {
            float t = dir_y;
            return (1.0f - t) * sky_horizon + t * sky_zenith;
        } else {
            float t = -dir_y;
            return (1.0f - t) * sky_horizon + t * ground_nadir;
        }
    }

    // If surface is specular (Glass/Dielectric or Metal/Mirror), trace reflected/refracted ray
    if (rec.material != nullptr) {
        const auto* dielectric = dynamic_cast<const Dielectric*>(rec.material);
        const auto* metal = dynamic_cast<const Metal*>(rec.material);
        if (dielectric != nullptr || metal != nullptr) {
            ScatterRecord srec;
            if (rec.material->scatter(ray, rec, srec)) {
                return srec.attenuation.cwiseProduct(
                    shade_direct_ray(srec.scattered, scene, options, depth + 1));
            }
        }
    }

    // Diffuse surface: evaluate direct Lambertian lighting with point lights and shadow rays
    Eigen::Vector3f albedo = rec.color;
    if (rec.material != nullptr) {
        const auto* lambert = dynamic_cast<const Lambertian*>(rec.material);
        if (lambert != nullptr) {
            albedo = lambert->albedo();
        } else {
            ScatterRecord srec;
            Ray dummy_ray(rec.point, rec.normal);
            if (rec.material->scatter(dummy_ray, rec, srec)) {
                albedo = srec.attenuation;
            }
        }
    }

    Eigen::Vector3f col = compute_direct_lighting(rec.point, rec.normal, albedo, scene.point_lights,
                                                  scene.hittables, options);

    // Distance-based atmospheric haze for distant plane hits (smooth exponential blend into
    // horizon)
    if (rec.t > 15.0f) {
        const Eigen::Vector3f sky_horizon(0.24f, 0.28f, 0.35f);
        float fog = 1.0f - std::exp(-0.04f * (rec.t - 15.0f));
        col = (1.0f - fog) * col + fog * sky_horizon;
    }

    return col;
}

}  // namespace

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
                pixel_col += shade_direct_ray(ray, scene, options, 0);
            }

            pixel_col /= static_cast<float>(samples_per_pixel);
            buffer.set_pixel(x, y, pixel_col);
        }
    }
}

}  // namespace mfad
