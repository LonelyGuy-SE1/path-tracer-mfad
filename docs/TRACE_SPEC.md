# MFAD Trace Format Specification

This document defines the schema, serialization semantics, and binary blob storage format for the MFAD trace verification pipeline across C++ (`trace/`) and Python (`prep/`, `viewer/`).

---

## 1. Trace Record Schema

A trace is a chronological sequence of linear algebra stage records emitted during preprocessing, mathematical stage execution, or test verification.

Each record is a JSON object with the following schema:

```json
{
  "stage": "<stage_name>",
  "name": "<record_identifier>",
  "shape": [<dim0>, <dim1>, ...],
  "data": <inline_array_or_binary_reference>,
  "note": "<human_readable_description>",
  "checks": {
    "<check_name>": <boolean_or_numeric_metric>
  }
}
```

### Field Definitions

1. **`stage`** (`string`, required):
   - The mathematical stage or subsystem generating the record.
   - Standard values: `"basis"`, `"transforms"`, `"projection"`, `"eigen"`, `"mesh_checks"`, `"svd_denoise"`, `"lsq_pose"`, `"dummy"`.

2. **`name`** (`string`, required):
   - A unique, descriptive key identifying this record within its stage invocation (e.g., `"duff_sample_0"`, `"rot_x_90"`, `"project_center"`).

3. **`shape`** (`array of integer`, required):
   - Tensor dimensions. For a scalar or 1D vector: `[N]`. For a 2D matrix: `[rows, cols]`.

4. **`data`** (`array of numbers` | `object with binary offset`, required):
   - **Small Matrices / Vectors (<= 10,000 elements)**:
     - Stored inline as nested JSON arrays of floating-point numbers (`double` precision).
   - **Large Arrays / Heavy Meshes (> 10,000 elements)**:
     - Stored out-of-line in a paired binary file (`<trace_name>.bin`).
     - JSON data entry contains:
       ```json
       {
         "$ref": "binary",
         "offset": 1048576,
         "length": 480000,
         "dtype": "float64"
       }
       ```

5. **`note`** (`string`, optional):
   - Mathematical notes, algorithm citations, or documentation (e.g., `"Duff et al. (2017) orthonormal basis [T, B, N]"`).

6. **`checks`** (`map<string, any>`, optional):
   - Key-value dictionary containing numerical invariants and assertion metrics.
   - Examples:
     - `"QtQ_error"`: `double` ($\|Q^T Q - I\|_\infty$)
     - `"QtQ_is_identity"`: `boolean`
     - `"det"`: `double` ($\det(Q)$)
     - `"det_is_one"`: `boolean`
     - `"passes_isometry"`: `boolean`

---

## 2. Example Record: Orthonormal Rotation Frame

```json
{
  "stage": "transforms",
  "name": "rot_z_90",
  "shape": [4, 4],
  "data": [
    [0.0, -1.0, 0.0, 0.0],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  0.0, 1.0, 0.0],
    [0.0,  0.0, 0.0, 1.0]
  ],
  "note": "4x4 transformation matrix analysis: rotate_z",
  "checks": {
    "QtQ_error": 0.0,
    "QtQ_is_identity": true,
    "det": 1.0,
    "det_is_one": true,
    "is_rotation": true,
    "passes_isometry": true,
    "transform_type": "rotate_z"
  }
}
```

---

## 3. Example Record: Out-of-Line Binary Mesh Geometry

```json
{
  "stage": "eigen",
  "name": "mesh_vertices_cube",
  "shape": [3, 24000],
  "data": {
    "$ref": "binary",
    "offset": 0,
    "length": 576000,
    "dtype": "float64"
  },
  "note": "Preprocessed mesh vertex matrix after welding and sliver removal",
  "checks": {
    "vertex_count": 24000,
    "rank": 3,
    "sliver_count": 0
  }
}
```

---

## 4. Binary File Layout

When out-of-line storage is used, binary blobs are appended sequentially to a companion `.bin` file:
- **Endianness**: Little-endian (standard x86-64 / ARM64).
- **Alignment**: 64-bit aligned (8 bytes).
- **Format**: Raw contiguous C-order or Fortran-order float64 array bytes matching NumPy / Eigen memory buffers.
