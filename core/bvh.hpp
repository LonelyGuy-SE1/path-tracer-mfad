#pragma once

#include "aabb.hpp"
#include "hittable.hpp"

#include <algorithm>
#include <memory>
#include <vector>

namespace mfad {

/**
 * @brief Bounding Volume Hierarchy node for O(log N) ray-scene intersection.
 * Uses SAH-inspired median split on the longest axis.
 */
class BVHNode : public Hittable {
public:
    BVHNode() = default;

    /**
     * @brief Builds a BVH from a list of hittable objects.
     */
    static std::shared_ptr<BVHNode> build(std::vector<std::shared_ptr<Hittable>>& objects,
                                          size_t start, size_t end);

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    AABB bounding_box() const override { return box_; }

private:
    std::shared_ptr<Hittable> left_;
    std::shared_ptr<Hittable> right_;
    AABB box_;
};

}  // namespace mfad
