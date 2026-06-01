#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>

namespace py = pybind11;
using namespace std;

vector<vector<double>> dmap(int N, const vector<int>& row, const vector<int>& col, int b = 256) {
    int n_b = ceil(1.0*N / b);
    vector<vector<double>> C(n_b, vector<double>(n_b, 0.0));
    int tnum = omp_get_max_threads();
    int chunk_size = ceil(1.0 * row.size() / (tnum * 8));
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int k = 0; k < row.size(); ++k) {
        int i = row[k];
        int j = col[k];
        // cout<<"debugging "<<k<<"row "<<row.size()<<endl; 
        // cout<<i/b<<" "<<j/b<<" "<<n_b<<endl;
        C[i / b][j / b] += 1;
    }
    // cout << "C: " << C.size() << " " << C[0].size() << endl;
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < n_b; ++i) {
        for (int j = 0; j < n_b; ++j) {
            C[i][j] /= (b * b);
        }
    }
    return C;
}

vector<vector<double>> edm(const vector<vector<double>>& dm_A, const vector<vector<double>>& dm_B, int n_b, int b = 256) {
    vector<vector<double>> result(n_b, vector<double>(n_b, 0.0));
    assert(dm_A.size() == dm_B.size() && dm_A[0].size() == dm_B[0].size());
    int tnum = omp_get_max_threads();
    int chunk_size = ceil(1.0 * n_b / (tnum * 8));
    #pragma omp parallel for schedule(dynamic, chunk_size) collapse(2)
    for (int i = 0; i < n_b; ++i) {
        for (int k = 0; k < n_b; ++k) {
            double sp1 = dm_A[i][k];
            if (sp1 == 0) {
                continue;
            }
            for (int j = 0; j < n_b; ++j) {
                double sp2 = dm_B[k][j];
                if (sp2 == 0) {
                    continue;
                }
                double tmp1 = 1 - pow(1 - sp1 * sp2, b);
                double tmp2 = result[i][j];
                result[i][j]= tmp2 + tmp1 - tmp1 * tmp2;
            }
        }
    }
    return result;
}

double sum_matrix(const std::vector<std::vector<double>>& matrix) {
    double sum = 0.0;
    for (const auto& row : matrix) {
        sum += std::accumulate(row.begin(), row.end(), 0.0);
    }
    return sum;
}

vector<vector<double>> calculate_result(int N, const vector<int>& row_A, const vector<int>& col_A, const vector<int>& row_B, const vector<int>& col_B, int b = 256) {
    // Calculate result_A using dmap
    vector<vector<double>> result_A = dmap(N, row_A, col_A, b);
    // cout << "result_A: " << result_A.size() << " " << result_A[0].size() << endl;
    // Calculate result_B using dmap
    vector<vector<double>> result_B = dmap(N, row_B, col_B, b);
    // cout << "result_B: " << result_B.size() << " " << result_B[0].size() << endl;

    // Calculate the final result using edm
    int n_b = ceil(1.0*N / b);
    vector<vector<double>> result = edm(result_A, result_B, n_b, b);
    // cout << "result: " << result.size() << " " << result[0].size() << endl;

    return result;
}

py::array_t<double> calculate_result_wrapper(int N, py::array_t<int> row_A, py::array_t<int> col_A, py::array_t<int> row_B, py::array_t<int> col_B, int b = 256) {
    // Convert NumPy arrays to std::vector
    std::vector<int> row_vec_A(row_A.size());
    std::vector<int> col_vec_A(col_A.size());
    std::vector<int> row_vec_B(row_B.size());
    std::vector<int> col_vec_B(col_B.size());
    std::memcpy(row_vec_A.data(), row_A.data(), row_A.size() * sizeof(int));
    std::memcpy(col_vec_A.data(), col_A.data(), col_A.size() * sizeof(int));
    std::memcpy(row_vec_B.data(), row_B.data(), row_B.size() * sizeof(int));
    std::memcpy(col_vec_B.data(), col_B.data(), col_B.size() * sizeof(int));

    // Call calculate_result function
    vector<vector<double>> result = calculate_result(N, row_vec_A, col_vec_A, row_vec_B, col_vec_B, b);

    // Convert result to NumPy array
    py::array_t<double> result_array({result.size(), result[0].size()});
    auto result_buf = result_array.request();
    double *result_ptr = static_cast<double *>(result_buf.ptr);
    for (size_t i = 0; i < result.size(); ++i) {
        std::memcpy(result_ptr + i * result[0].size(), result[i].data(), result[0].size() * sizeof(double));
    }
    return result_array;
}
py::array_t<double> edm_wrapper(py::array_t<double> dm_A, py::array_t<double> dm_B, int n_b, int b = 256) {
    // Convert NumPy arrays to std::vector
    std::vector<std::vector<double>> vec_dm_A(dm_A.shape(0), std::vector<double>(dm_A.shape(1)));
    std::vector<std::vector<double>> vec_dm_B(dm_B.shape(0), std::vector<double>(dm_B.shape(1)));
    std::memcpy(vec_dm_A.data()->data(), dm_A.data(), dm_A.size() * sizeof(double));
    std::memcpy(vec_dm_B.data()->data(), dm_B.data(), dm_B.size() * sizeof(double));

    // Call edm function
    vector<vector<double>> result = edm(vec_dm_A, vec_dm_B, n_b, b);

    // Convert result to NumPy array
    py::array_t<double> result_array({result.size(), result[0].size()});
    auto result_buf = result_array.request();
    double *result_ptr = static_cast<double *>(result_buf.ptr);
    for (size_t i = 0; i < result.size(); ++i) {
        std::memcpy(result_ptr + i * result[0].size(), result[i].data(), result[0].size() * sizeof(double));
    }
    return result_array;
}

py::array_t<double> dmap_wrapper(int N, py::array_t<int> row, py::array_t<int> col, int b = 256) {
    // Convert NumPy arrays to std::vector
    std::vector<int> row_vec(row.size());
    std::vector<int> col_vec(col.size());
    std::memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
    std::memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));

    // Call dmap function
    vector<vector<double>> result = dmap(N, row_vec, col_vec, b);

    // Convert result to NumPy array
    py::array_t<double> result_array({result.size(), result[0].size()});
    auto result_buf = result_array.request();
    double *result_ptr = static_cast<double *>(result_buf.ptr);
    for (size_t i = 0; i < result.size(); ++i) {
        std::memcpy(result_ptr + i * result[0].size(), result[i].data(), result[0].size() * sizeof(double));
    }
    return result_array;
}
vector<double> dm_chain(int N, const vector<pair<vector<int>, vector<int>>>& index_list, int b = 256) {
    int p = index_list.size();
    int n_b = ceil(1.0*N / b);
    long double dim = 1.0*N*N;
    vector<double> result(p, 0.0);
    vector<vector<double>> left(n_b, vector<double>(n_b, 0.0));

    left = dmap(N, index_list[0].first, index_list[0].second, b);
    result[0] = index_list[0].first.size() / dim;

    for (int i = 1; i < p; i++) {
        vector<vector<double>> right(n_b, vector<double>(n_b, 0.0));
        vector<vector<double>> tmp(n_b, vector<double>(n_b, 0.0));
        right = dmap(N, index_list[i].first, index_list[i].second, b);
        tmp = edm(left, right, n_b, b);
        left = tmp;
        result[i] = sum_matrix(tmp)*256*256;
    }
    return result;
}

py::array_t<double> dm_chain_wrapper(int N, py::list index_list, int b = 256) {
    // Convert Python list of tuples to vector of pairs of vectors
    vector<pair<vector<int>, vector<int>>> vec_index_list;
    for (auto item : index_list) {
        auto tuple = item.cast<py::tuple>();
        py::array_t<int> row = tuple[0].cast<py::array_t<int>>();
        py::array_t<int> col = tuple[1].cast<py::array_t<int>>();
        vector<int> row_vec(row.size());
        vector<int> col_vec(col.size());
        std::memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
        std::memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));
        vec_index_list.push_back(make_pair(row_vec, col_vec));
    }

    // Call dm_chain function
    vector<double> result = dm_chain(N, vec_index_list, b);
    // Convert result to NumPy array
    py::array_t<double> result_array(result.size());
    auto result_buf = result_array.request();
    double *result_ptr = static_cast<double *>(result_buf.ptr);
    std::memcpy(result_ptr, result.data(), result.size() * sizeof(double));
    return result_array;
}




PYBIND11_MODULE(DensityMap, m) {
    m.doc() = "pybind11 densitymap plugin"; // optional module docstring
    m.def("dm", &calculate_result_wrapper, "A function that calculates the result",
          py::arg("N"), py::arg("row_A"), py::arg("col_A"), py::arg("row_B"), py::arg("col_B"), py::arg("b") = 256);
    m.def("edm", &edm_wrapper, "A function that calculates the result",
          py::arg("dm_A"), py::arg("dm_B"), py::arg("n_b"), py::arg("b") = 256);
    m.def("dmap", &dmap_wrapper, "A function that calculates the density map",
          py::arg("N"), py::arg("row"), py::arg("col"), py::arg("b") = 256);
    m.def("dm_chain", &dm_chain_wrapper, "A function that calculates the result of a chain of density maps",
            py::arg("N"), py::arg("index_list"), py::arg("b") = 256);
}
