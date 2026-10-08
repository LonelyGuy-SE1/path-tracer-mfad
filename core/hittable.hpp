#pragma once

#include "hit_record.hpp"
#include "ray.hpp"
#include "aabb.hpp"

#include <memory>
#include <vector>

namespace mfad {

class BVHNode;

/**
 * @brief Base interface for all geometric objects in the scene.
 */
class Hittable {
public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
    virtual AABB bounding_box() const = 0;
};

/**
 * @brief Container holding multiple Hittable objects.
 */
class HittableList : public Hittable {
public:
    HittableList() = default;

    void add(std::shared_ptr<Hittable> object) { objects_.push_back(std::move(object)); }
    void clear() { objects_.clear(); }

    void build_bvh();

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;

    AABB bounding_box() const override {
        if (objects_.empty()) return AABB();
        AABB temp_box = objects_[0]->bounding_box();
        for (size_t i = 1; i < objects_.size(); i++) {
            temp_box = AABB::surrounding_box(temp_box, objects_[i]->bounding_box());
        }
        return temp_box;
    }

    const std::vector<std::shared_ptr<Hittable>>& objects() const { return objects_; }

private:
    std::vector<std::shared_ptr<Hittable>> objects_;
    std::shared_ptr<BVHNode> bvh_root_;
};

}  // namespace mfad
