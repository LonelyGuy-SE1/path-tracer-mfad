#include "bvh.hpp"
#include <algorithm>
#include <iostream>

namespace mfad {

std::shared_ptr<BVHNode> BVHNode::build(std::vector<std::shared_ptr<Hittable>>& objects,
                                         size_t start, size_t end) {
    auto node = std::make_shared<BVHNode>();
    size_t count = end - start;

    if (count == 1) {
        node->left_ = objects[start];
        node->right_ = objects[start];
        node->box_ = objects[start]->bounding_box();
        return node;
    }

    if (count == 2) {
        node->left_ = objects[start];
        node->right_ = objects[start + 1];
        node->box_ = AABB::surrounding_box(objects[start]->bounding_box(),
                                           objects[start + 1]->bounding_box());
        return node;
    }

    // Compute bounding box of all objects in range
    AABB total_box = objects[start]->bounding_box();
    for (size_t i = start + 1; i < end; ++i) {
        total_box = AABB::surrounding_box(total_box, objects[i]->bounding_box());
    }
    int axis = total_box.longest_axis();

    // Sort by centroid on the chosen axis
    std::sort(objects.begin() + start, objects.begin() + end,
              [axis](const std::shared_ptr<Hittable>& a, const std::shared_ptr<Hittable>& b) {
                  return a->bounding_box().centroid()[axis] < b->bounding_box().centroid()[axis];
              });

    size_t mid = start + count / 2;
    node->left_ = build(objects, start, mid);
    node->right_ = build(objects, mid, end);
    node->box_ = AABB::surrounding_box(node->left_->bounding_box(), node->right_->bounding_box());
    return node;
}

bool BVHNode::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    if (!box_.hit(r, t_min, t_max)) {
        return false;
    }

    bool hit_left = left_->hit(r, t_min, t_max, rec);
    bool hit_right = right_->hit(r, t_min, hit_left ? rec.t : t_max, rec);
    return hit_left || hit_right;
}

}  // namespace mfad
