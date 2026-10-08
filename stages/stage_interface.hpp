#pragma once

#include "trace.hpp"

#include <string>

namespace mfad {

/**
 * @brief Standard result wrapper returned by MFAD mathematical stages.
 *
 * @tparam T Type of the computed mathematical value (e.g., Matrix4d, Vector3d, double).
 */
template <typename T> struct StageResult {
    T value;
    bool success{true};
    std::string stage_name;
    std::string note;

    StageResult() = default;
    StageResult(T val, bool succ = true, std::string stage = "", std::string n = "")
        : value(std::move(val)), success(succ), stage_name(std::move(stage)), note(std::move(n)) {}
};

/**
 * @brief Dummy stage implementing the standard stage signature:
 * StageResult<T> stage_<name>(input, Trace& trace)
 * Used to verify compiler compatibility and pipeline recording (Issue #4).
 */
inline StageResult<double> stage_dummy(double input, Trace& trace,
                                       const std::string& item_name = "dummy_record") {
    double output = input * 2.0;
    Eigen::VectorXd vec(2);
    vec << input, output;

    trace.record_vector("dummy", item_name, vec, "Testing stage interface",
                        {{"input_valid", true}, {"doubled", output == input * 2.0}});

    return StageResult<double>(output, true, "dummy", "Successfully executed dummy stage");
}

}  // namespace mfad
