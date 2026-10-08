#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace mfad {

/**
 * @brief 2D HDR Float Image Buffer storing linear RGB colors.
 * Supports exporting to 8-bit sRGB PNG format using stb_image_write.
 */
class ImageBuffer {
public:
    ImageBuffer(int width, int height);

    int width() const { return width_; }
    int height() const { return height_; }

    void set_pixel(int x, int y, const Eigen::Vector3f& color);
    Eigen::Vector3f get_pixel(int x, int y) const;

    /**
     * @brief Writes image to PNG file with gamma correction.
     * @param filepath Destination path.
     * @param gamma Gamma exponent (standard sRGB is 2.2).
     */
    bool write_png(const std::string& filepath, float gamma = 2.2f) const;

    /**
     * @brief Writes image to Radiance HDR (.hdr) format storing 32-bit linear floating point RGB
     * (Issue #30).
     * @param filepath Destination path (.hdr).
     */
    bool write_hdr(const std::string& filepath) const;

    const std::vector<Eigen::Vector3f>& data() const { return pixels_; }

private:
    int width_;
    int height_;
    std::vector<Eigen::Vector3f> pixels_;
};

}  // namespace mfad
