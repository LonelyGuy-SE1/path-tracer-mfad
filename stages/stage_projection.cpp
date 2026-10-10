#include "stage_projection.hpp"

#include <algorithm>
#include <cmath>

namespace mfad {

StageResult<Eigen::Vector3d> stage_reflect(const Eigen::Vector3d& v, const Eigen::Vector3d& n,
                                           Trace* trace, const std::string& item_name) {
    Eigen::Vector3d unit_n = n.normalized();
    Eigen::Vector3d r = v - 2.0 * v.dot(unit_n) * unit_n;

    // Linear algebra checks
    double v_len = v.norm();
    double r_len = r.norm();
    double len_error = std::abs(v_len - r_len);
    bool len_preserved = (len_error < 1e-12);

    double v_dot_n = v.dot(unit_n);
    double r_dot_n = r.dot(unit_n);
    double angle_error = std::abs(v_dot_n + r_dot_n);
    bool angle_preserved = (angle_error < 1e-12);

    if (trace != nullptr) {
        trace->record_vector("projection", item_name, r,
                             "Orthogonal reflection across normal: r = v - 2(v.n)n",
                             {{"length_preserved", len_preserved},
                              {"length_error", len_error},
                              {"angle_preserved", angle_preserved},
                              {"angle_error", angle_error},
                              {"v_dot_n", v_dot_n},
                              {"r_dot_n", r_dot_n}});
    }

    return StageResult<Eigen::Vector3d>(r, true, "projection", "Reflection vector calculated");
}

StageResult<double> stage_diffuse_term(const Eigen::Vector3d& normal,
                                       const Eigen::Vector3d& light_dir, Trace* trace,
                                       const std::string& item_name) {
    Eigen::Vector3d unit_n = normal.normalized();
    Eigen::Vector3d unit_l = light_dir.normalized();

    double n_dot_l = unit_n.dot(unit_l);
    double diffuse = std::max(0.0, n_dot_l);

    bool in_range = (diffuse >= 0.0 && diffuse <= 1.0 + 1e-12);
    bool illuminated = (diffuse > 0.0);

    if (trace != nullptr) {
        Eigen::VectorXd record_val(1);
        record_val << diffuse;
        trace->record_vector("projection", item_name, record_val,
                             "Lambertian diffuse cosine factor: max(0, n.l)",
                             {{"n_dot_l", n_dot_l},
                              {"diffuse", diffuse},
                              {"in_range", in_range},
                              {"illuminated", illuminated}});
    }

    return StageResult<double>(diffuse, true, "projection", "Diffuse term calculated");
}

StageResult<ShadowRayResult> stage_shadow_direction(const Eigen::Vector3d& surface_point,
                                                    const Eigen::Vector3d& light_pos, double eps,
                                                    Trace* trace, const std::string& item_name) {
    Eigen::Vector3d to_light = light_pos - surface_point;
    double dist = to_light.norm();
    Eigen::Vector3d dir = (dist > 1e-12) ? to_light / dist : Eigen::Vector3d(0.0, 1.0, 0.0);
    Eigen::Vector3d offset_origin = surface_point + eps * dir;

    bool valid_dist = (dist > 1e-12);
    bool valid_dir = (std::abs(dir.norm() - 1.0) < 1e-12);

    ShadowRayResult result;
    result.direction = dir;
    result.distance = dist;
    result.shadow_origin = offset_origin;

    if (trace != nullptr) {
        trace->record_vector("projection", item_name, dir,
                             "Shadow ray direction and offset test vector",
                             {{"distance", dist},
                              {"valid_dist", valid_dist},
                              {"valid_dir", valid_dir},
                              {"offset_epsilon", eps}});
    }

    return StageResult<ShadowRayResult>(result, true, "projection", "Shadow ray test complete");
}

StageResult<ProjectResult> stage_project_image_plane(const Eigen::Vector3d& point_view_space,
                                                     double vfov_deg, double aspect_ratio,
                                                     int width, int height, double near_plane,
                                                     double far_plane, Trace* trace,
                                                     const std::string& item_name) {
    const double pi = 3.14159265358979323846;
    double theta = vfov_deg * pi / 180.0;
    double tan_half_fov = std::tan(theta / 2.0);

    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    P(0, 0) = 1.0 / (aspect_ratio * tan_half_fov);
    P(1, 1) = 1.0 / tan_half_fov;
    P(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    P(2, 3) = -(2.0 * far_plane * near_plane) / (far_plane - near_plane);
    P(3, 2) = -1.0;

    Eigen::Vector4d p_homo(point_view_space.x(), point_view_space.y(), point_view_space.z(), 1.0);
    Eigen::Vector4d clip = P * p_homo;

    ProjectResult res;
    res.projection_matrix = P;

    if (std::abs(clip.w()) > 1e-12) {
        res.ndc = clip.head<3>() / clip.w();
    } else {
        res.ndc = Eigen::Vector3d::Zero();
    }

    // Check NDC bounds [-1, 1]^3
    res.in_frustum = (res.ndc.x() >= -1.0 && res.ndc.x() <= 1.0 && res.ndc.y() >= -1.0 &&
                      res.ndc.y() <= 1.0 && res.ndc.z() >= -1.0 && res.ndc.z() <= 1.0);

    // Map NDC to pixel coordinates [0, width] x [0, height]
    double px = 0.5 * (res.ndc.x() + 1.0) * static_cast<double>(width);
    // Y-flipped so (0, 0) is top-left in screen pixels
    double py = 0.5 * (1.0 - res.ndc.y()) * static_cast<double>(height);
    res.pixel = Eigen::Vector2d(px, py);

    if (trace != nullptr) {
        trace->record_matrix("projection", item_name, P,
                             "4x4 Perspective projection matrix onto image plane",
                             {{"vfov_deg", vfov_deg},
                              {"aspect_ratio", aspect_ratio},
                              {"near_plane", near_plane},
                              {"far_plane", far_plane},
                              {"in_frustum", res.in_frustum},
                              {"ndc_x", res.ndc.x()},
                              {"ndc_y", res.ndc.y()},
                              {"ndc_z", res.ndc.z()},
                              {"pixel_x", px},
                              {"pixel_y", py}});
    }

    return StageResult<ProjectResult>(res, true, "projection", "Image plane projection complete");
}

}  // namespace mfad
