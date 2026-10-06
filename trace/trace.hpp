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

class Trace {
public:
    Trace() = default;

    void record_matrix(const std::string& stage, const std::string& name,
                       const Eigen::MatrixXd& mat, const std::string& note = "",
                       const std::map<std::string, nlohmann::json>& checks = {}) {
        TraceRecord rec;
        rec.stage = stage;
        rec.name = name;
        rec.shape = {static_cast<int>(mat.rows()), static_cast<int>(mat.cols())};
        rec.note = note;
        rec.checks = checks;

        std::vector<std::vector<double>> values(mat.rows(), std::vector<double>(mat.cols()));
        for (int r = 0; r < mat.rows(); ++r) {
            for (int c = 0; c < mat.cols(); ++c) {
                values[r][c] = mat(r, c);
            }
        }
        rec.data = values;
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

        std::vector<double> values(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            values[i] = vec(i);
        }
        rec.data = values;
        records_.push_back(std::move(rec));
    }

    bool save_json(const std::string& filepath) const {
        nlohmann::json root = nlohmann::json::array();
        for (const auto& rec : records_) {
            nlohmann::json j;
            j["stage"] = rec.stage;
            j["name"] = rec.name;
            j["shape"] = rec.shape;
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
    void clear() { records_.clear(); }

private:
    std::vector<TraceRecord> records_;
};

}  // namespace mfad
