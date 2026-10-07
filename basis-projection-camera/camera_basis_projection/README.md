# Basis + Projection Camera

This is a small camera module for `path-tracer-mfad` that uses the project's Basis and Projection stages.

## Files

- `stages/stage_camera.hpp` — Camera interface and ray result.
- `stages/stage_camera.cpp` — builds the camera frame with `stage_basis_gram_schmidt()` and projects the image-plane offset with `stage_projection_subspace()`.
- `tests/test_camera.cpp` — basic center/left/right/top ray checks.
- `CMake_camera.txt` — CMake block to add the camera test.

## Integration

Copy the two `stage_camera.*` files into the repository's `stages/` directory and `test_camera.cpp` into `tests/`. Then copy the contents of `CMake_camera.txt` into the repository `CMakeLists.txt` after the existing stage tests.

The implementation assumes the existing Basis API from this branch:

```cpp
BasisResult stage_basis_gram_schmidt(const Eigen::Vector3d& normal,
                                     const Eigen::Vector3d& guide,
                                     Trace* trace = nullptr,
                                     const std::string& item_name = ...);
```

and uses `BasisResult::Q`, whose columns are `[T, B, N]` as documented by the current Basis stage.
