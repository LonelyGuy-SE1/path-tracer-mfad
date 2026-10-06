#pragma once

#include "hit_record.hpp"
#include "ray.hpp"

#include <memory>
#include <vector>

namespace mfad {

/**
 * @brief Base interface for all geometric objects in the scene.
 */
class Hittable {
public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

/**
 * @brief Container holding multiple Hittable objects.
 */
class HittableList : public Hittable {
public:
    HittableList() = default;

    void add(std::shared_ptr<Hittable> object) { objects_.push_back(std::move(object)); }
    void clear() { objects_.clear(); }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (const auto& object : objects_) {
            if (object->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    const std::vector<std::shared_ptr<Hittable>>& objects() const { return objects_; }

private:
    std::vector<std::shared_ptr<Hittable>> objects_;
};

}  // namespace mfad
