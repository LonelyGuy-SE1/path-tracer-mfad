#include "image_buffer.hpp"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace mfad {

ImageBuffer::ImageBuffer(int width, int height)
    : width_(width), height_(height), pixels_(width * height, Eigen::Vector3f::Zero()) {}

void ImageBuffer::set_pixel(int x, int y, const Eigen::Vector3f& color) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        pixels_[y * width_ + x] = color;
    }
}

Eigen::Vector3f ImageBuffer::get_pixel(int x, int y) const {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        return pixels_[y * width_ + x];
    }
    return Eigen::Vector3f::Zero();
}

bool ImageBuffer::write_png(const std::string& filepath, float gamma) const {
    std::vector<unsigned char> bytes(width_ * height_ * 3);
    const float inv_gamma = 1.0f / gamma;

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Eigen::Vector3f& c = pixels_[y * width_ + x];

            // Gamma correction: c^(1/gamma), clamped to [0.0, 1.0]
            float r = std::clamp(std::pow(std::max(0.0f, c.x()), inv_gamma), 0.0f, 1.0f);
            float g = std::clamp(std::pow(std::max(0.0f, c.y()), inv_gamma), 0.0f, 1.0f);
            float b = std::clamp(std::pow(std::max(0.0f, c.z()), inv_gamma), 0.0f, 1.0f);

            int idx = (y * width_ + x) * 3;
            bytes[idx + 0] = static_cast<unsigned char>(r * 255.0f);
            bytes[idx + 1] = static_cast<unsigned char>(g * 255.0f);
            bytes[idx + 2] = static_cast<unsigned char>(b * 255.0f);
        }
    }

    return stbi_write_png(filepath.c_str(), width_, height_, 3, bytes.data(), width_ * 3) != 0;
}

}  // namespace mfad
