#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <omp.h>

namespace py = pybind11;
using namespace std;

struct Graph {
    vector<vector<int>> rev_A;  // rev_A[j] = src nodes i with A[i][j]=1
    vector<vector<int>> rev_B;  // rev_B[j] = mid nodes i with B[i][j]=1
    int N;
};

Graph construct_graph(int N, const vector<int>& row_A, const vector<int>& col_A, const vector<int>& row_B, const vector<int>& col_B) {
    Graph graph;
    graph.N = N;
    graph.rev_A.resize(N);
    graph.rev_B.resize(N);
    for (size_t i = 0; i < row_A.size(); ++i) {
        graph.rev_A[col_A[i]].push_back(row_A[i]);
    }
    for (size_t i = 0; i < row_B.size(); ++i) {
        graph.rev_B[col_B[i]].push_back(row_B[i]);
    }
    return graph;
}

vector<double> layer_graph(int N, const vector<int>& row_A, const vector<int>& col_A, const vector<int>& row_B, const vector<int>& col_B, int r = 16) {
    vector<double> res(N, 0.0);
    Graph graph = construct_graph(N, row_A, col_A, row_B, col_B);
    vector<vector<double>> left(N, vector<double>(r));
    vector<vector<double>> mid(N, vector<double>(r, 1e10));
    vector<vector<double>> right(N, vector<double>(r, 1e10));

    // 初始化节点特征
    random_device rd;
    mt19937 gen(rd());
    double lambda = 1.0;
    exponential_distribution<double> distribution(lambda);
    int tnum = omp_get_max_threads();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < r; ++j) {
            left[i][j] = distribution(gen);
        }
    }

    // 更新 mid 节点特征（按 dst 并行，每个 mid[j] 仅由一线程写）
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        for (int i : graph.rev_A[j]) {
            for (int k = 0; k < r; ++k) {
                mid[j][k] = min(mid[j][k], left[i][k]);
            }
        }
    }

    // 更新 right 节点特征（按 dst 并行，每个 right[j] 仅由一线程写）
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        for (int i : graph.rev_B[j]) {
            for (int k = 0; k < r; ++k) {
                right[j][k] = min(right[j][k], mid[i][k]);
            }
        }
    }

    // 计算结果
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double sum = accumulate(right[i].begin(), right[i].end(), 0.0);
        res[i] = (r - 1) / sum;
        if (res[i] < 0) {
            res[i] = 0;
        }
    }

    return res;
}

py::array_t<double> layer_graph_wrapper(int N, py::array_t<int> row_A, py::array_t<int> col_A, py::array_t<int> row_B, py::array_t<int> col_B, int r = 16) {
    // 将 NumPy 数组转换为 std::vector
    vector<int> row_vec_A(row_A.size());
    vector<int> col_vec_A(col_A.size());
    vector<int> row_vec_B(row_B.size());
    vector<int> col_vec_B(col_B.size());
    memcpy(row_vec_A.data(), row_A.data(), row_A.size() * sizeof(int));
    memcpy(col_vec_A.data(), col_A.data(), col_A.size() * sizeof(int));
    memcpy(row_vec_B.data(), row_B.data(), row_B.size() * sizeof(int));
    memcpy(col_vec_B.data(), col_B.data(), col_B.size() * sizeof(int));

    // 调用 layer_graph 函数
    vector<double> result = layer_graph(N, row_vec_A, col_vec_A, row_vec_B, col_vec_B, r);

    // 将结果转换为 NumPy 数组
    py::array_t<double> result_array(result.size(), result.data());
    return result_array;
}

struct GraphChain {
    vector<vector<vector<int>>> rev_adj_layers;  // rev_adj_layers[layer][dst] = src nodes
    int N;
};

GraphChain construct_graph_chain(int N, const vector<pair<vector<int>, vector<int>>>& index_list) {
    GraphChain graph_chain;
    graph_chain.N = N;
    graph_chain.rev_adj_layers.resize(index_list.size());
    for (size_t i = 0; i < index_list.size(); ++i) {
        graph_chain.rev_adj_layers[i].resize(N);
        for (size_t j = 0; j < index_list[i].first.size(); ++j) {
            graph_chain.rev_adj_layers[i][index_list[i].second[j]].push_back(index_list[i].first[j]);
        }
    }
    return graph_chain;
}

vector<double> layer_graph_chain(int N, const vector<pair<vector<int>, vector<int>>>& index_list, int r = 16) {
    vector<double> result;
    GraphChain graph_chain = construct_graph_chain(N, index_list);
    vector<vector<vector<double>>> layers(index_list.size() + 1, vector<vector<double>>(N, vector<double>(r, 1e10)));
    long double dim = 1.0* N * N;
    double lambda = 1.0;
    // 初始化节点特征
    random_device rd;
    mt19937 gen(rd());
    exponential_distribution<double> distribution(lambda);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < r; ++j) {
            layers[0][i][j] = distribution(gen);
        }
    }
    int tnum = omp_get_max_threads();
    // 更新各层节点特征
    for (size_t i = 0; i < index_list.size(); ++i) {
        #pragma omp parallel for
        for (int k = 0; k < N; ++k) {
            for (int j : graph_chain.rev_adj_layers[i][k]) {
                for (int l = 0; l < r; ++l) {
                    layers[i + 1][k][l] = min(layers[i + 1][k][l], layers[i][j][l]);
                }
            }
        }
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < N; ++j) {
            double tmp_sum = accumulate(layers[i + 1][j].begin(), layers[i + 1][j].end(), 0.0);
            double res = (r-1) / tmp_sum;
            if (res >= 0) {
                sum += res;
            }
        }
        result.push_back(sum / dim);
    }
    return result;
}
std::vector<int> numpy_to_vector_int(py::array_t<int> array) {
    py::buffer_info info = array.request();
    int* ptr = static_cast<int*>(info.ptr);
    return std::vector<int>(ptr, ptr + info.size);
}


py::array_t<double> layer_graph_chain_wrapper(int N, py::list index_list_py, int r = 16) {
    // 将 Python 列表转换为 C++ vector<pair<vector<int>, vector<int>>>
    vector<pair<vector<int>, vector<int>>> index_list;
    for (auto item : index_list_py) {
        py::tuple t = py::cast<py::tuple>(item);

        vector<int> first = numpy_to_vector_int(py::cast<py::array_t<int>>(t[0]));
        vector<int> second = numpy_to_vector_int(py::cast<py::array_t<int>>(t[1]));

        index_list.push_back(make_pair(first, second));
    }

    // 调用 layer_graph_chain 函数
    vector<double> result = layer_graph_chain(N, index_list, r);

    // 将结果转换为 NumPy 数组
    py::array_t<double> result_array(result.size(), result.data());
    return result_array;
}

vector<double> layer_graph_chain_seed(int N,
                                      const vector<pair<vector<int>, vector<int>>>& index_list,
                                      int r,
                                      uint64_t seed) {
    vector<double> result;
    GraphChain graph_chain = construct_graph_chain(N, index_list);
    vector<vector<vector<double>>> layers(index_list.size() + 1, vector<vector<double>>(N, vector<double>(r, 1e10)));
    long double dim = 1.0* N * N;
    double lambda = 1.0;
    mt19937_64 gen(seed);
    exponential_distribution<double> distribution(lambda);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < r; ++j) {
            layers[0][i][j] = distribution(gen);
        }
    }
    int tnum = omp_get_max_threads();
    // 更新各层节点特征
    for (size_t i = 0; i < index_list.size(); ++i) {
        #pragma omp parallel for
        for (int k = 0; k < N; ++k) {
            for (int j : graph_chain.rev_adj_layers[i][k]) {
                for (int l = 0; l < r; ++l) {
                    layers[i + 1][k][l] = min(layers[i + 1][k][l], layers[i][j][l]);
                }
            }
        }
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < N; ++j) {
            double tmp_sum = accumulate(layers[i + 1][j].begin(), layers[i + 1][j].end(), 0.0);
            double res = (r-1) / tmp_sum;
            if (res >= 0) {
                sum += res;
            }
        }
        result.push_back(sum / dim);
    }
    return result;
}

py::array_t<double> layer_graph_chain_seed_wrapper(int N, py::list index_list_py, int r, uint64_t seed) {
    vector<pair<vector<int>, vector<int>>> index_list;
    for (auto item : index_list_py) {
        py::tuple t = py::cast<py::tuple>(item);

        vector<int> first = numpy_to_vector_int(py::cast<py::array_t<int>>(t[0]));
        vector<int> second = numpy_to_vector_int(py::cast<py::array_t<int>>(t[1]));

        index_list.push_back(make_pair(first, second));
    }

    vector<double> result = layer_graph_chain_seed(N, index_list, r, seed);

    py::array_t<double> result_array(result.size(), result.data());
    return result_array;
}


PYBIND11_MODULE(LG_EXP1, m) {
    m.doc() = "pybind11 layer_graph plugin"; // optional module docstring
    m.def("layer_graph", &layer_graph_wrapper, "A function that computes the layer graph",
          py::arg("N"), py::arg("row_A"), py::arg("col_A"), py::arg("row_B"), py::arg("col_B"), py::arg("r") = 16);
    m.def("layer_graph_chain", &layer_graph_chain_wrapper, "A function that computes the layer graph chain",
            py::arg("N"), py::arg("index_list"), py::arg("r") = 16);
    m.def("layer_graph_chain_seed", &layer_graph_chain_seed_wrapper,
          "layer_graph_chain with explicit RNG seed for initial exponential features",
          py::arg("N"), py::arg("index_list"), py::arg("r"), py::arg("seed"));
}