#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace mfad {

struct TraceRecord {
    std::string stage;
    std::string name;
    std::vector<int> shape;
    nlohmann::json data;
    std::string note;
    std::map<std::string, nlohmann::json> checks;
};

/**
 * @brief C++ Trace writer implementing the MFAD Trace format spec with JSON and binary sidecar
 * support (Issue #5).
 */
class Trace {
public:
    static constexpr size_t DEFAULT_INLINE_LIMIT = 4096;

    explicit Trace(size_t inline_limit = DEFAULT_INLINE_LIMIT) : inline_limit_(inline_limit) {}

    void record_matrix(const std::string& stage, const std::string& name,
                       const Eigen::MatrixXd& mat, const std::string& note = "",
                       const std::map<std::string, nlohmann::json>& checks = {}) {
        TraceRecord rec;
        rec.stage = stage;
        rec.name = name;
        rec.shape = {static_cast<int>(mat.rows()), static_cast<int>(mat.cols())};
        rec.note = note;
        rec.checks = checks;

        size_t total_elements = static_cast<size_t>(mat.rows() * mat.cols());
        if (total_elements <= inline_limit_) {
            std::vector<std::vector<double>> values(mat.rows(), std::vector<double>(mat.cols()));
            for (int r = 0; r < mat.rows(); ++r) {
                for (int c = 0; c < mat.cols(); ++c) {
                    values[r][c] = mat(r, c);
                }
            }
            rec.data = values;
        } else {
            // Write to binary buffer
            size_t offset = bin_buffer_.size();
            size_t nbytes = total_elements * sizeof(double);
            // Append row-major contiguous data
            for (int r = 0; r < mat.rows(); ++r) {
                for (int c = 0; c < mat.cols(); ++c) {
                    double val = mat(r, c);
                    const char* bytes = reinterpret_cast<const char*>(&val);
                    bin_buffer_.insert(bin_buffer_.end(), bytes, bytes + sizeof(double));
                }
            }
            nlohmann::json bin_ref;
            bin_ref["$bin"] = {{"offset", offset}, {"nbytes", nbytes}, {"dtype", "float64"}};
            rec.data = bin_ref;
        }

        records_.push_back(std::move(rec));
    }

    void record_vector(const std::string& stage, const std::string& name,
                       const Eigen::VectorXd& vec, const std::string& note = "",
                       const std::map<std::string, nlohmann::json>& checks = {}) {
        TraceRecord rec;
        rec.stage = stage;
        rec.name = name;
        rec.shape = {static_cast<int>(vec.size())};
        rec.note = note;
        rec.checks = checks;

        size_t total_elements = static_cast<size_t>(vec.size());
        if (total_elements <= inline_limit_) {
            std::vector<double> values(vec.size());
            for (int i = 0; i < vec.size(); ++i) {
                values[i] = vec(i);
            }
            rec.data = values;
        } else {
            size_t offset = bin_buffer_.size();
            size_t nbytes = total_elements * sizeof(double);
            for (int i = 0; i < vec.size(); ++i) {
                double val = vec(i);
                const char* bytes = reinterpret_cast<const char*>(&val);
                bin_buffer_.insert(bin_buffer_.end(), bytes, bytes + sizeof(double));
            }
            nlohmann::json bin_ref;
            bin_ref["$bin"] = {{"offset", offset}, {"nbytes", nbytes}, {"dtype", "float64"}};
            rec.data = bin_ref;
        }

        records_.push_back(std::move(rec));
    }

    bool save_json(const std::string& filepath) {
        std::string bin_filename;
        if (!bin_buffer_.empty()) {
            std::string bin_path = filepath;
            size_t dot_pos = bin_path.find_last_of('.');
            size_t slash_pos = bin_path.find_last_of("/\\");
            if (dot_pos != std::string::npos &&
                (slash_pos == std::string::npos || dot_pos > slash_pos)) {
                bin_path = bin_path.substr(0, dot_pos) + ".bin";
            } else {
                bin_path += ".bin";
            }

            // Extract just the filename for relative reference
            slash_pos = bin_path.find_last_of("/\\");
            bin_filename =
                (slash_pos != std::string::npos) ? bin_path.substr(slash_pos + 1) : bin_path;

            std::ofstream bin_out(bin_path, std::ios::binary);
            if (!bin_out.is_open()) {
                return false;
            }
            bin_out.write(bin_buffer_.data(), bin_buffer_.size());
        }

        nlohmann::json root = nlohmann::json::array();
        for (auto& rec : records_) {
            nlohmann::json j;
            j["stage"] = rec.stage;
            j["name"] = rec.name;
            j["shape"] = rec.shape;
            if (rec.data.is_object() && rec.data.contains("$bin") && !bin_filename.empty()) {
                rec.data["$bin"]["file"] = bin_filename;
            }
            j["data"] = rec.data;
            j["note"] = rec.note;
            j["checks"] = rec.checks;
            root.push_back(j);
        }

        std::ofstream out(filepath);
        if (!out.is_open()) {
            return false;
        }
        out << root.dump(2);
        return true;
    }

    const std::vector<TraceRecord>& records() const { return records_; }
    const std::vector<char>& binary_buffer() const { return bin_buffer_; }
    void clear() {
        records_.clear();
        bin_buffer_.clear();
    }

private:
    size_t inline_limit_;
    std::vector<TraceRecord> records_;
    std::vector<char> bin_buffer_;
};

}  // namespace mfad
