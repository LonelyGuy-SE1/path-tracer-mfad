#pragma once

#include "scene.hpp"

#include <string>

namespace mfad {

/**
 * @brief High-level loader parsing JSON scene specifications into MFAD matrix representations
 * and ray-traceable primitives (Issue #20).
 */
class SceneLoader {
public:
    /**
     * @brief Loads a scene from a JSON file.
     * Reads camera, materials, meshes, quadrics, and lights into linear algebra matrices.
     * Logs the parsed object count and primitive statistics.
     *
     * @param filepath Absolute or relative path to the scene JSON file.
     * @param aspect_ratio Viewport aspect ratio (width / height) for camera configuration.
     * @return Scene fully populated with matrices, materials, and hittables.
     */
    static Scene load_from_json(const std::string& filepath, double aspect_ratio = 16.0 / 9.0);
};

}  // namespace mfad
