#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <iostream>

namespace py = pybind11;
using namespace std;

struct SummaryStat {
    int max_row;
    int max_col;
    int n_half_row;
    int n_half_col;
    int n_oh_row;
    int n_oh_col;
};

tuple<vector<long double>, vector<long double>, vector<long double>, vector<long double>, SummaryStat> mnc_sketch(int N, const vector<int>& row, const vector<int>& col) {
    vector<long double> h_r(N, 0.0), h_c(N, 0.0);
    vector<long double> h_er, h_ec;
    SummaryStat summary_stat = {0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < row.size(); ++i) {
        h_r[row[i]] += 1;
        h_c[col[i]] += 1;
    }
    if (*max_element(h_r.begin(), h_r.end()) > 1 || *max_element(h_c.begin(), h_c.end()) > 1) {
        h_er.resize(N, 0.0);
        h_ec.resize(N, 0.0);
        for (size_t i = 0; i < row.size(); ++i) {
            if (h_c[col[i]] == 1) {
                h_er[row[i]] += 1;
            }
            if (h_r[row[i]] == 1) {
                h_ec[col[i]] += 1;
            }
        }
    }
    summary_stat.max_row = *max_element(h_r.begin(), h_r.end());
    summary_stat.max_col = *max_element(h_c.begin(), h_c.end());
    summary_stat.n_half_row = count_if(h_r.begin(), h_r.end(), [N](long double x) { return x >= 0.5 * N; });
    summary_stat.n_half_col = count_if(h_c.begin(), h_c.end(), [N](long double x) { return x >= 0.5 * N; });
    summary_stat.n_oh_row = count_if(h_r.begin(), h_r.end(), [](long double x) { return x == 1; });
    summary_stat.n_oh_col = count_if(h_c.begin(), h_c.end(), [](long double x) { return x == 1; });

    return make_tuple(h_r, h_c, h_er, h_ec, summary_stat);
}

std::tuple<long double, vector<int>, vector<int>> mnc(int N, const vector<int>& row_A, const vector<int>& col_A, const vector<int>& row_B, const vector<int>& col_B) {
    auto [A_h_r, A_h_c, A_h_er, A_h_ec, A_summary_stat] = mnc_sketch(N, row_A, col_A);
    auto [B_h_r, B_h_c, B_h_er, B_h_ec, B_summary_stat] = mnc_sketch(N, row_B, col_B);
    long double nnz = 0;
    int sum_Aec = accumulate(A_h_ec.begin(), A_h_ec.end(), 0);
    int sum_Ber = accumulate(B_h_er.begin(), B_h_er.end(), 0);
    if (A_summary_stat.max_row <= 1 || B_summary_stat.max_col <= 1) {
        // cout << "case 1" << endl;
        for (int j = 0; j < N; ++j) {
            nnz += A_h_c[j] * B_h_r[j];
        }
    } else if (sum_Aec>=1 || sum_Ber>=1) {
        // cout << "case 2" << endl;
        long double mnOut = (count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x != 0; }) - A_summary_stat.n_oh_row) *
                       (count_if(B_h_c.begin(), B_h_c.end(), [](long double x) { return x != 0; }) - B_summary_stat.n_oh_col);
        long double spOutrest = 0;
        for (int j = 0; j < N; ++j) {
            nnz += A_h_ec[j] * B_h_r[j];
            nnz += (A_h_c[j] - A_h_ec[j]) * B_h_er[j];
            long double lsp = (A_h_c[j] - A_h_ec[j]) * (B_h_r[j] - B_h_er[j]) / mnOut;
            spOutrest = spOutrest + lsp - lsp * spOutrest;
        }
        nnz += mnOut * spOutrest;
        // cout<<"finished case 2"<<endl;
    } else {
        // cout << "case 3" << endl;
        long double mnOut = count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x != 0; }) *
                       count_if(B_h_c.begin(), B_h_c.end(), [](long double x) { return x != 0; });
        long double spOut = 0;
        for (int j = 0; j < N; ++j) {
            long double lsp = A_h_c[j] * B_h_r[j] / mnOut;
            spOut = spOut + lsp - lsp * spOut;
        }
        nnz = mnOut * spOut;
    }
    nnz = max(nnz, static_cast<long double>(A_summary_stat.n_half_row * B_summary_stat.n_half_col));

    
    vector<int> C_h_r(N, 0), C_h_c(N, 0);
    long double sum_A_h_r = accumulate(A_h_r.begin(), A_h_r.end(), 0.0);
    long double sum_B_h_c = accumulate(B_h_c.begin(), B_h_c.end(), 0.0);
    // cout<<"sum_A_h_r: "<<sum_A_h_r<<endl;
    // cout<<"sum_B_h_c: "<<sum_B_h_c<<endl;
    // cout<<"nnz: "<<nnz<<endl;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        C_h_r[i] = (int)round(A_h_r[i] * nnz / sum_A_h_r);
        C_h_c[i] = (int)round(B_h_c[i] * nnz / sum_B_h_c);
        // if (A_h_r[i] >= 1 || B_h_c[i] >= 1 ) {
        //     cout<<"A_h_r[i]: "<<A_h_r[i]<<endl;
        //     cout<<"B_h_c[i]: "<<B_h_c[i]<<endl;
        //     cout<<"C_h_r[i]: "<<C_h_r[i]<<endl;
        //     cout<<"C_h_c[i]: "<<C_h_c[i]<<endl;
        // }
        
    }
    return make_tuple(nnz, C_h_r, C_h_c);

    
}

std::tuple<long double, vector<int>, vector<int>>  mnc_chain_iter(int N, vector<int> &A_h_r, vector<int> &A_h_c, const vector<int>& row_B, const vector<int>& col_B, bool return_sketch = false) {
    auto [B_h_r, B_h_c, B_h_er, B_h_ec, B_summary_stat] = mnc_sketch(N, row_B, col_B);
    long double max_A_h_r = *max_element(A_h_r.begin(), A_h_r.end());
    long double nnz = 0;
    long double sum_Ber = accumulate(B_h_er.begin(), B_h_er.end(), 0);
    if (max_A_h_r <= 1 || B_summary_stat.max_col <= 1) {
        // cout << "case 1" << endl;
        for (int j = 0; j < N; ++j) {
            nnz += A_h_c[j] * B_h_r[j];
        }
        // return nnz;
    } else if (sum_Ber >= 1) {
        // cout << "case 2 iter " << endl;
        int nnz_A = count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x != 0; });
        int nnz_B = count_if(B_h_c.begin(), B_h_c.end(), [](long double x) { return x != 0; });
        int A_n_oh_row = count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x == 1; });
        // cout<<"nnz_A: "<<nnz_A<<" one hot A "<<A_n_oh_row <<endl;
        // cout<<"nnz_B: "<<nnz_B<<" one hot B "<<B_summary_stat.n_oh_col<<endl;
        long double mnOut = (count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x != 0; }) - A_n_oh_row) *
                       (count_if(B_h_c.begin(), B_h_c.end(), [](long double x) { return x != 0; }) - B_summary_stat.n_oh_col);
        long double spOutrest = 0;
        // cout<<"mnOut: "<<mnOut<<endl;
        for (int j = 0; j < N; ++j) {
            nnz += (A_h_c[j] ) * B_h_er[j];
            long double lsp = (A_h_c[j]) * (B_h_r[j] - B_h_er[j]) / mnOut;
            spOutrest = spOutrest + lsp - lsp * spOutrest;
        }
        nnz += mnOut * spOutrest;
        // return nnz;
    } else {
        // cout << "case 3" << endl;
        long double mnOut = count_if(A_h_r.begin(), A_h_r.end(), [](long double x) { return x != 0; }) *
                       count_if(B_h_c.begin(), B_h_c.end(), [](long double x) { return x != 0; });
        long double spOut = 0;
        for (int j = 0; j < N; ++j) {
            long double lsp = A_h_c[j] * B_h_r[j] / mnOut;
            spOut = spOut + lsp - lsp * spOut;
        }
        nnz = mnOut * spOut;
        // return nnz;
    }
    //sketch propagate
    vector<int> C_h_r(N, 0), C_h_c(N, 0);
    long double sum_A_h_r = accumulate(A_h_r.begin(), A_h_r.end(), 0.0);
    long double sum_B_h_c = accumulate(B_h_c.begin(), B_h_c.end(), 0.0);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        C_h_r[i] = (int)round(A_h_r[i] * nnz / sum_A_h_r);
        C_h_c[i] = (int)round(B_h_c[i] * nnz / sum_B_h_c);

    }
    // cout << "Sum of C_h_r: " << accumulate(C_h_r.begin(), C_h_r.end(), 0) << endl;
    // cout << "Sum of C_h_c: " << accumulate(C_h_c.begin(), C_h_c.end(), 0) << endl;
    // cout << "nnz: " << nnz << endl;
    return make_tuple(nnz, C_h_r, C_h_c);

}

vector<long double> mnc_chain(int N, const vector<pair<vector<int>, vector<int>>>& index_list) {
    vector<int> C_h_r(N, 0), C_h_c(N, 0);
    int p = index_list.size();
    long double nnz = 0;
    vector<long double> nnz_vec(p, 0.0);
    vector<int> it_C_h_r(N, 0), it_C_h_c(N, 0);
    for (int i = 0; i < p; i++) {
        if (i == 0) {
            nnz = index_list[i].first.size();
            nnz_vec[i] = nnz;
        } 
        if (i == 1){
            auto [nnz, new_C_h_r, new_C_h_c] = mnc(N, index_list[i-1].first, index_list[i-1].second, index_list[i].first, index_list[i].second);
            C_h_r = new_C_h_r;
            C_h_c = new_C_h_c;
            nnz_vec[i] = nnz;
        }
        if (i > 1) {
            // cout << "Sum of C_h_r: " << accumulate(C_h_r.begin(), C_h_r.end(), 0) << endl;
            // cout << "Sum of C_h_c: " << accumulate(C_h_c.begin(), C_h_c.end(), 0) << endl;
            auto [nnz, new_C_h_r, new_C_h_c] = mnc_chain_iter(N, C_h_r, C_h_c, index_list[i].first, index_list[i].second);
            nnz_vec[i] = nnz;
            C_h_r = new_C_h_r;
            C_h_c = new_C_h_c;
        }
    }
    return nnz_vec;
}

py::tuple mnc_sketch_wrapper(int N, py::array_t<int> row, py::array_t<int> col) {
    // 将 NumPy 数组转换为 std::vector
    vector<int> row_vec(row.size());
    vector<int> col_vec(col.size());
    memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
    memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));

    // 调用 mnc_sketch 函数
    auto [h_r, h_c, h_er, h_ec, summary_stat] = mnc_sketch(N, row_vec, col_vec);

    // 将结果转换为 NumPy 数组和字典
    py::array_t<long double> h_r_array(h_r.size(), h_r.data());
    py::array_t<long double> h_c_array(h_c.size(), h_c.data());
    py::array_t<long double> h_er_array(h_er.size(), h_er.data());
    py::array_t<long double> h_ec_array(h_ec.size(), h_ec.data());
    py::dict summary_stat_dict;
    summary_stat_dict["max_row"] = summary_stat.max_row;
    summary_stat_dict["max_col"] = summary_stat.max_col;
    summary_stat_dict["n_half_row"] = summary_stat.n_half_row;
    summary_stat_dict["n_half_col"] = summary_stat.n_half_col;
    summary_stat_dict["n_oh_row"] = summary_stat.n_oh_row;
    summary_stat_dict["n_oh_col"] = summary_stat.n_oh_col;

    return py::make_tuple(h_r_array, h_c_array, h_er_array, h_ec_array, summary_stat_dict);
}

py::tuple mnc_wrapper(int N, py::array_t<int> row_A, py::array_t<int> col_A, py::array_t<int> row_B, py::array_t<int> col_B) {
    // 将 NumPy 数组转换为 std::vector
    vector<int> row_A_vec(row_A.size());
    vector<int> col_A_vec(col_A.size());
    vector<int> row_B_vec(row_B.size());
    vector<int> col_B_vec(col_B.size());
    memcpy(row_A_vec.data(), row_A.data(), row_A.size() * sizeof(int));
    memcpy(col_A_vec.data(), col_A.data(), col_A.size() * sizeof(int));
    memcpy(row_B_vec.data(), row_B.data(), row_B.size() * sizeof(int));
    memcpy(col_B_vec.data(), col_B.data(), col_B.size() * sizeof(int));

    // 调用 mnc 函数
    auto [nnz, C_h_r, C_h_c] = mnc(N, row_A_vec, col_A_vec, row_B_vec, col_B_vec);
    // cout<<"all good for mnc"<<endl;
    // cout << "Shape of C_h_r: " << C_h_r.size() << endl;
    // cout << "Shape of C_h_c: " << C_h_c.size() << endl;
    py::array_t<int> C_h_r_array(C_h_r.size(), 0);
    py::array_t<int> C_h_c_array(C_h_c.size(), 0);
    memcpy(C_h_r_array.mutable_data(), C_h_r.data(), C_h_r.size() * sizeof(int));
    memcpy(C_h_c_array.mutable_data(), C_h_c.data(), C_h_c.size() * sizeof(int));

    // 将结果转换为 NumPy 数组
    return py::make_tuple(nnz, C_h_r_array, C_h_c_array);
}
py::array_t<long double> mnc_chain_wrapper(int N, py::list index_list) {
    vector<pair<vector<int>, vector<int>>> index_vector;
    for (auto item : index_list) {
        py::tuple tuple_item = py::cast<py::tuple>(item);
        py::array_t<int> row = py::cast<py::array_t<int>>(tuple_item[0]);
        py::array_t<int> col = py::cast<py::array_t<int>>(tuple_item[1]);
        vector<int> row_vec(row.size());
        vector<int> col_vec(col.size());
        memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
        memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));
        index_vector.push_back(make_pair(row_vec, col_vec));
    }

    vector<long double> nnz_vec = mnc_chain(N, index_vector);
    py::array_t<long double> nnz_array(nnz_vec.size(), nnz_vec.data());
    return nnz_array;
}

PYBIND11_MODULE(mnc, m) {
    m.doc() = "pybind11 mnc plugin"; // optional module docstring
    m.def("mnc", &mnc_wrapper, "A function that computes the MNC",
          py::arg("N"), py::arg("row_A"), py::arg("col_A"), py::arg("row_B"), py::arg("col_B"));
    m.def("mnc_sketch", &mnc_sketch_wrapper, "A function that computes the MNC sketch",
          py::arg("N"), py::arg("row"), py::arg("col"));
    m.def("mnc_chain", &mnc_chain_wrapper, "A function that computes the MNC chain",
          py::arg("N"), py::arg("index_list"));
}