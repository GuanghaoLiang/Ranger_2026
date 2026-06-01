#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

enum class SmcmBackend { CPU, Kokkos };

enum class SmcmPreset {
    Gt,
    GtCore,
    GtKk,
    GtKkCore,
};

struct SmcmRunConfig {
    int p = 0;
    std::string dataset;
    SmcmBackend backend = SmcmBackend::CPU;
    bool use_core = false;
    std::string multiply_order = "all";
    std::string matrix_root;
    std::string metapath_root;
    std::string output_root;
    int max_metapaths = 0;
};

struct MetapathJob {
    std::string mp;
};

struct SmcmOutputPaths {
    std::string output;
    std::string log;
};

using CooMatrix = std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>;
using CooList = std::vector<CooMatrix>;
using ShapeList = std::vector<std::pair<int, int>>;

void apply_env_defaults(SmcmRunConfig& config);
SmcmRunConfig preset_to_config(SmcmPreset preset);
SmcmOutputPaths resolve_output_paths(const SmcmRunConfig& config);
int run_smcm_benchmark(const SmcmRunConfig& config);
int smcm_run_with_preset(int argc, char* argv[], SmcmPreset preset);
void print_smcm_run_usage(const char* program);
int parse_smcm_run_args(int argc, char* argv[], SmcmRunConfig& config);

template <typename IT, typename NT>
inline void read_coo_matrix(const std::string& filename, IT& rows, IT& cols, IT& nnz,
                            std::vector<IT>& row_indices, std::vector<IT>& col_indices,
                            std::vector<NT>& values) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    std::string line;
    if (!std::getline(fin, line)) {
        throw std::runtime_error("文件为空: " + filename);
    }

    std::istringstream header(line);
    IT n = 0;
    if (!(header >> n)) {
        throw std::runtime_error("第一行读取失败: " + filename);
    }

    rows = n;
    cols = n;
    if (!std::getline(fin, line)) {
        throw std::runtime_error("缺少 row 行: " + filename);
    }
    std::istringstream row_stream(line);
    row_indices.clear();
    IT value = 0;
    while (row_stream >> value) {
        row_indices.push_back(value);
    }
    if (!std::getline(fin, line)) {
        throw std::runtime_error("缺少 col 行: " + filename);
    }
    std::istringstream col_stream(line);
    col_indices.clear();
    while (col_stream >> value) {
        col_indices.push_back(value);
    }
    nnz = static_cast<IT>(col_indices.size());
    if (static_cast<IT>(row_indices.size()) == n + 1 && !row_indices.empty() &&
        row_indices.back() == nnz) {
        std::vector<IT> expanded_rows;
        expanded_rows.reserve(nnz);
        for (IT r = 0; r < n; ++r) {
            const IT start = row_indices[r];
            const IT end = row_indices[r + 1];
            for (IT k = start; k < end; ++k) {
                expanded_rows.push_back(r);
            }
        }
        row_indices = std::move(expanded_rows);
    } else if (static_cast<IT>(row_indices.size()) != nnz) {
        throw std::runtime_error("row/col 长度不匹配: " + filename);
    }
    values.assign(static_cast<size_t>(nnz), static_cast<NT>(1));
}

inline std::vector<std::string> split_mp_tokens(const std::string& input) {
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, '-')) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

inline std::vector<std::string> matrix_names_from_mp(const std::string& input, int p) {
    const std::vector<std::string> tokens = split_mp_tokens(input);
    if (static_cast<int>(tokens.size()) != p + 1) {
        throw std::runtime_error(
            "compressed metapath expects p+1 tokens for p=" + std::to_string(p) + ": " + input);
    }
    std::vector<std::string> matrices;
    for (int j = 0; j < p; ++j) {
        matrices.push_back(tokens[j] + "-" + tokens[j + 1]);
    }
    return matrices;
}

template <typename IT, typename NT>
inline std::pair<CooList, ShapeList> read_mat_by_mp(const std::string& mp,
                                                    const std::string& filepath,
                                                    int p) {
    std::vector<std::string> matrices_name = matrix_names_from_mp(mp, p);
    CooList index_list;
    ShapeList shape_list;

    for (const auto& mat : matrices_name) {
        std::vector<IT> row_indices, col_indices;
        std::vector<NT> values;
        IT rows = 0, cols = 0, nnz = 0;
        std::string filename = filepath + '/' + mat + ".txt";
        read_coo_matrix(filename, rows, cols, nnz, row_indices, col_indices, values);
        index_list.push_back(std::make_tuple(row_indices, col_indices, values));
        shape_list.push_back(std::make_pair(rows, cols));
    }
    return std::make_pair(index_list, shape_list);
}

inline std::vector<MetapathJob> read_metapath_jobs(const std::string& filename) {
    std::vector<MetapathJob> jobs;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open metapath file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string mp;
        if (!(iss >> mp) || mp.empty() || mp[0] == '#') {
            continue;
        }
        jobs.push_back({mp});
    }
    return jobs;
}

inline std::pair<int, int> getFirstAndLastFieldOfLastLine(const std::string& filename, bool& resume) {
    std::ifstream file(filename);
    std::string line;
    std::string lastLine;

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        resume = false;
        return {0, 0};
    }

    while (std::getline(file, line)) {
        if (!line.empty()) {
            lastLine = line;
        }
    }

    file.close();

    if (lastLine.empty()) {
        std::cerr << "文件为空或最后一行为空" << std::endl;
        resume = false;
        return {0, 0};
    }

    std::istringstream iss(lastLine);
    std::vector<std::string> fields;
    std::string field;

    while (iss >> field) {
        fields.push_back(field);
    }

    if (fields.empty()) {
        std::cerr << "最后一行没有有效字段" << std::endl;
        resume = false;
        return {0, 0};
    }

    resume = true;
    return {std::stoi(fields.front()), std::stoi(fields.back())};
}
