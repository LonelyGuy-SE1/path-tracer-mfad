#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mfad {

class ImageBuffer {
public:
    ImageBuffer(int width, int height);

    int width() const { return width_; }
    int height() const { return height_; }

    void set_pixel(int x, int y, const Eigen::Vector3f& color);
    Eigen::Vector3f get_pixel(int x, int y) const;

    bool write_png(const std::string& filepath, float gamma = 2.2f, bool apply_aces = true) const;

    bool write_hdr(const std::string& filepath) const;

    const std::vector<Eigen::Vector3f>& data() const { return pixels_; }

private:
    int width_;
    int height_;
    std::vector<Eigen::Vector3f> pixels_;
};

}  // namespace mfad
