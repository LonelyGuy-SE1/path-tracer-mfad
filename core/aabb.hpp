#pragma once

#include "ray.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

namespace mfad {

/**
 * @brief Axis-Aligned Bounding Box for BVH acceleration.
 */
class AABB {
public:
    Eigen::Vector3f min_pt{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    Eigen::Vector3f max_pt{std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};

    AABB() = default;
    AABB(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
        : min_pt(a.cwiseMin(b)), max_pt(a.cwiseMax(b)) {}

    /**
     * @brief Fast slab-based ray-AABB intersection test.
     */
    bool hit(const Ray& r, float t_min, float t_max) const {
        for (int i = 0; i < 3; ++i) {
            float inv_d = 1.0f / r.direction[i];
            float t0 = (min_pt[i] - r.origin[i]) * inv_d;
            float t1 = (max_pt[i] - r.origin[i]) * inv_d;
            if (inv_d < 0.0f)
                std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max < t_min)
                return false;
        }
        return true;
    }

    /**
     * @brief Merges this AABB with another to produce the surrounding box.
     */
    static AABB surrounding_box(const AABB& a, const AABB& b) {
        return AABB(a.min_pt.cwiseMin(b.min_pt), a.max_pt.cwiseMax(b.max_pt));
    }

    /**
     * @brief Returns the longest axis index (0=x, 1=y, 2=z) for split heuristic.
     */
    int longest_axis() const {
        Eigen::Vector3f d = max_pt - min_pt;
        if (d.x() > d.y() && d.x() > d.z())
            return 0;
        if (d.y() > d.z())
            return 1;
        return 2;
    }

    Eigen::Vector3f centroid() const { return 0.5f * (min_pt + max_pt); }
};

}  // namespace mfad
