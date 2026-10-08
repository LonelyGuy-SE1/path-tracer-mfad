#include "hittable.hpp"
#include "bvh.hpp"

namespace mfad {

void HittableList::build_bvh() {
    if (!objects_.empty()) {
        bvh_root_ = BVHNode::build(objects_, 0, objects_.size());
    }
}

bool HittableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    if (bvh_root_) {
        return bvh_root_->hit(r, t_min, t_max, rec);
    }
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

}  // namespace mfad
