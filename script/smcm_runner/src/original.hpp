#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>
#include "mkl.h"
#include "mkl_spblas.h"
using namespace std;


std::set<int> get_row_csr(sparse_matrix_t A_csr, int n_row, int n_col) {
    auto t0 = std::chrono::high_resolution_clock::now();
    sparse_index_base_t csr_indexing;
    int *rows_start, *rows_end, *col_indices;
    double *csr_values;
    MKL_INT info = mkl_sparse_d_export_csr(
        A_csr, &csr_indexing, &n_row, &n_col, &rows_start, &rows_end, &col_indices, &csr_values
    );
    if (info != 0) {
        throw std::runtime_error("Failed to export CSR matrix: " + std::to_string(info));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    // cout << "**********Time taken to export CSR matrix: " << elapsed.count() << " seconds" << endl;

    std::set<int> row_indices;
    int n_threads = omp_get_max_threads();
    std::vector<std::set<int>> local_sets(n_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n_row; ++i) {
            if (rows_start[i] != rows_end[i]) {
                local_sets[tid].insert(i);
            }
        }
    }

    // 合并所有线程的本地集合
    for (int t = 0; t < n_threads; ++t) {
        row_indices.insert(local_sets[t].begin(), local_sets[t].end());
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    // cout << "**********Time taken to find non-zero rows: " << elapsed.count() << " seconds" << endl;
    
    return row_indices;
}

std::set<int> get_row_indices(pair<vector<int>, vector<int>> index_pair, int n_row, int n_col) {
    auto t0 = std::chrono::high_resolution_clock::now();
    long nnz = index_pair.first.size();
    std::vector<double> ones(nnz, 1.0);
    sparse_matrix_t A_csr, A_coo;
    mkl_sparse_d_create_coo(
            &A_coo,
            SPARSE_INDEX_BASE_ZERO,
            n_row, n_col,
            nnz,
            std::get<0>(index_pair).data(),
            std::get<1>(index_pair).data(),
            ones.data()
        );
    mkl_sparse_convert_csr(A_coo, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    sparse_index_base_t csr_indexing;
    int *rows_start, *rows_end, *col_indices;
    double *csr_values;
    MKL_INT info = mkl_sparse_d_export_csr(
        A_csr, &csr_indexing, &n_row, &n_col, &rows_start, &rows_end, &col_indices, &csr_values
    );
    if (info != 0) {
        throw std::runtime_error("Failed to export CSR matrix: " + std::to_string(info));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    // cout << "**********Time taken to export CSR matrix: " << elapsed.count() << " seconds" << endl;
    std::set<int> row_indices;
    int n_threads = omp_get_max_threads();
    std::vector<std::set<int>> local_sets(n_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n_row; ++i) {
            if (rows_start[i] != rows_end[i]) {
                local_sets[tid].insert(i);
            }
        }
    }

    // 合并所有线程的本地集合
    for (int t = 0; t < n_threads; ++t) {
        row_indices.insert(local_sets[t].begin(), local_sets[t].end());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    // cout << "**********Time taken to find non-zero rows: " << elapsed.count() << " seconds" << endl;
    return row_indices;
}

std::set<int> get_col_indices(pair<vector<int>, vector<int>> index_pair, int n_row, int n_col) {
    auto t0 = std::chrono::high_resolution_clock::now();
    long nnz = index_pair.first.size();
    std::vector<double> ones(nnz, 1.0);
    sparse_matrix_t A_csr, A_coo;
    mkl_sparse_d_create_coo(
            &A_coo,
            SPARSE_INDEX_BASE_ZERO,
            n_row, n_col,
            nnz,
            std::get<1>(index_pair).data(),
            std::get<0>(index_pair).data(),
            ones.data()
        );
    mkl_sparse_convert_csr(A_coo, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    sparse_index_base_t csr_indexing;
    int *cols_start, *cols_end, *row_indices;
    double *csr_values;
    MKL_INT info = mkl_sparse_d_export_csr(
        A_csr, &csr_indexing, &n_row, &n_col, &cols_start, &cols_end, &row_indices, &csr_values
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    // cout << "**********Time taken to export CSC matrix: " << elapsed.count() << " seconds" << endl;
    if (info != 0) {
        throw std::runtime_error("Failed to export CSC matrix: " + std::to_string(info));
    }
    std::set<int> col_indices;
    int n_threads = omp_get_max_threads();
    // printf("Number of threads: %d\n", n_threads);
    std::vector<std::set<int>> local_sets(n_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < n_col; ++i) {
            if (cols_start[i] != cols_end[i]) {
                local_sets[tid].insert(i);
            }
        }
    }

    // 合并所有线程的本地集合
    for (int t = 0; t < n_threads; ++t) {
        col_indices.insert(local_sets[t].begin(), local_sets[t].end());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    // cout << "**********Time taken to find non-zero columns: " << elapsed.count() << " seconds" << endl;
    return col_indices;
}
std::set<int> get_col_indices_set(pair<vector<int>, vector<int>> index_pair, int n_row, int n_col, set<int> right_set, int verbose = 1) {
    auto t0 = std::chrono::high_resolution_clock::now();
    long nnz = index_pair.first.size();
    std::vector<float> ones(nnz, 1.0);
    sparse_matrix_t A_csr, A_coo;
    mkl_sparse_s_create_coo(
            &A_coo,
            SPARSE_INDEX_BASE_ZERO,
            n_row, n_col,
            nnz,
            std::get<1>(index_pair).data(),
            std::get<0>(index_pair).data(),
            ones.data()
        );
    auto t05 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t05 - t0;
    // cout << "Time taken to create COO matrix: " << elapsed.count() << " seconds" << endl;
    // 转换 COO 到 CSR
    mkl_sparse_convert_csr(A_coo, SPARSE_OPERATION_NON_TRANSPOSE, &A_csr);
    sparse_index_base_t csr_indexing;
    int *cols_start, *cols_end, *row_indices;
    float *csr_values;
    MKL_INT info = mkl_sparse_s_export_csr(
        A_csr, &csr_indexing, &n_row, &n_col, &cols_start, &cols_end, &row_indices, &csr_values
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        elapsed = t1 - t05;
        cout << "Time taken to export CSC matrix: " << elapsed.count() << " seconds" << endl;
    }
    if (info != 0) {
        throw std::runtime_error("Failed to export CSC matrix: " + std::to_string(info));
    }
    std::set<int> col_indices;
    int n_right = right_set.size();
    int n_threads = omp_get_max_threads();
    // printf("Number of threads: %d\n", n_threads);
    std::vector<std::set<int>> local_sets(n_threads);
    std::vector<int> right_vec(right_set.begin(), right_set.end());
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < right_vec.size(); ++i) {
            int val = right_vec[i];
            if (cols_start[val] != cols_end[val]) {
                local_sets[tid].insert(val);
            }
        }
    }
    // 合并所有线程的本地集合
    for (int t = 0; t < n_threads; ++t) {
        col_indices.insert(local_sets[t].begin(), local_sets[t].end());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    if (verbose>0){
        elapsed = t2 - t1;
        cout << "**********Time taken to find non-zero columns: " << elapsed.count() << " seconds" << endl;
    }
    return col_indices;
}
std::set<int> get_col_indices_set_spmv(pair<vector<int>, vector<int>> index_pair, int n_row, int n_col, set<int> right_set) {
    auto t0 = std::chrono::high_resolution_clock::now();
    long nnz = index_pair.first.size();
    std::vector<double> ones(nnz, 1.0);
    sparse_matrix_t A_coo;
    mkl_sparse_d_create_coo(
            &A_coo,
            SPARSE_INDEX_BASE_ZERO,
            n_row, n_col,
            nnz,
            std::get<1>(index_pair).data(),
            std::get<0>(index_pair).data(),
            ones.data()
        );
    auto t05 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t05 - t0;
    cout << "Time taken to create COO matrix: " << elapsed.count() << " seconds" << endl;
    // spmv
    std::vector<double> x(n_row, 1.0);   // 全1向量
    std::vector<double> y;   // 输出结果

    sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
    matrix_descr descL;
    double alpha = 1.0, beta = 0.0;
    sparse_status_t status;

    status = mkl_sparse_d_mv(opA, alpha, A_coo, descL, x.data(), beta, y.data());
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Error in mkl_sparse_d_mv" << status << std::endl;
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed = t1 - t05;
    cout << "**********Time taken to spmv: " << elapsed.count() << " seconds" << endl;
    std::set<int> col_indices;
    int n_right = right_set.size();
    int n_threads = omp_get_max_threads();
    // printf("Number of threads: %d\n", n_threads);
    std::vector<std::set<int>> local_sets(n_threads);
    std::vector<int> right_vec(right_set.begin(), right_set.end());
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < right_vec.size(); ++i) {
            int val = right_vec[i];
            if (y[val] > 0.0) {
                local_sets[tid].insert(val);
            }
        }
    }
    // 合并所有线程的本地集合
    for (int t = 0; t < n_threads; ++t) {
        col_indices.insert(local_sets[t].begin(), local_sets[t].end());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    cout << "**********Time taken to find non-zero columns: " << elapsed.count() << " seconds" << endl;
    return col_indices;
}

// std::set<int> get_col_indices_set_seq(pair<vector<int>, vector<int>> index_pair, int n_row, int n_col, set<int> right_set) {
//     auto t0 = std::chrono::high_resolution_clock::now();
//     std::set<int> col_indices = set<int>(index_pair.second.begin(), index_pair.second.end());
//     auto t05 = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = t05 - t0;
//     cout << "Time taken to create set from vector: " << elapsed.count() << " seconds" << endl;
//     // spmv
//     int n_right = right_set.size();
//     int n_threads = omp_get_max_threads();
//     printf("Number of threads: %d\n", n_threads);
//     std::vector<std::set<int>> local_sets(n_threads);
//     std::vector<int> right_vec(right_set.begin(), right_set.end());
// #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         #pragma omp for
//         for (size_t i = 0; i < right_vec.size(); ++i) {
//             int val = right_vec[i];
//             if (y[val] > 0.0) {
//                 local_sets[tid].insert(val);
//             }
//         }
//     }
//     // 合并所有线程的本地集合
//     for (int t = 0; t < n_threads; ++t) {
//         col_indices.insert(local_sets[t].begin(), local_sets[t].end());
//     }
//     auto t2 = std::chrono::high_resolution_clock::now();
//     elapsed = t2 - t1;
//     cout << "**********Time taken to find non-zero columns and intersection: " << elapsed.count() << " seconds" << endl;
//     return col_indices;
// }

std::vector<pair<vector<int>, vector<int>>> cool(vector<pair<vector<int>, vector<int>>>& index_list, std::vector<std::pair<int, int>> shape_list) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;

        // set<int> left_set(left.begin(), left.end());

        // set<int> right_set(right.begin(), right.end());
        // auto t05 = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed = t05 - t0;
        // cout << "Time taken to convert vectors to sets: " << elapsed.count() << " seconds" << endl;
        // set<int> common;

        // set_intersection(left_set.begin(), left_set.end(), right_set.begin(), right_set.end(), inserter(common, common.begin()));
        
        // std::chrono::duration<double> elapsed = t1 - t0;
        // cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;

        // if (common.empty()) {
        //     cout << "No common element" << endl;
        //     return {};
        // }
        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        set<int> left_set2, right_set2, common2;
        
        left_set2 = get_col_indices(index_list[i], shape_list[i].first, shape_list[i].second);
        right_set2 = get_row_indices(index_list[i + 1], shape_list[i + 1].first, shape_list[i + 1].second);
        auto t1 = std::chrono::high_resolution_clock::now();
        set_intersection(right_set2.begin(), right_set2.end(), left_set2.begin(), left_set2.end(), inserter(common2, common2.begin()));
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t2 - t1;
        cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;

        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        elapsed = t3 - t2;
        cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        //////此步改为并发
        // remove_elements(index_list[i].first, left_mask);
        // remove_elements(index_list[i].second, left_mask);
        // remove_elements(index_list[i + 1].first, right_mask);
        // remove_elements(index_list[i + 1].second, right_mask);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].second, right_mask);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        elapsed = t5 - t4;
        cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        cout << "===================================" << endl;
    }
    return index_list;
}
std::vector<pair<vector<int>, vector<int>>> not_cool(vector<pair<vector<int>, vector<int>>>& index_list, std::vector<std::pair<int, int>> shape_list) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;

        set<int> left_set(left.begin(), left.end());

        set<int> right_set(right.begin(), right.end());
        auto t05 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t05 - t0;
        cout << "Time taken to convert vectors to sets: " << elapsed.count() << " seconds" << endl;
        set<int> common;

        set_intersection(left_set.begin(), left_set.end(), right_set.begin(), right_set.end(), inserter(common, common.begin()));
        auto t1 = std::chrono::high_resolution_clock::now();
        elapsed = t1 - t0;
        cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;

        if (common.empty()) {
            cout << "No common element" << endl;
            return {};
        }

        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        auto t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common.find(left[j]) != common.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common.find(right[j]) != common.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        elapsed = t3 - t2;
        cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        //////此步改为并发
        // remove_elements(index_list[i].first, left_mask);
        // remove_elements(index_list[i].second, left_mask);
        // remove_elements(index_list[i + 1].first, right_mask);
        // remove_elements(index_list[i + 1].second, right_mask);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].second, right_mask);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        elapsed = t5 - t4;
        cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        cout << "===================================" << endl;
    }
    return index_list;
}

std::vector<pair<vector<int>, vector<int>>> coolh(vector<pair<vector<int>, vector<int>>>& index_list, std::vector<std::pair<int, int>> shape_list, std::vector<set<int>> right_nrows, int verbose = 0) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;
        // cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        set<int> left_set2, right_set2, common2;
        right_set2 = right_nrows[i + 1];
        left_set2 = get_col_indices_set(index_list[i], shape_list[i].first, shape_list[i].second, right_set2, verbose = 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        // set_intersection(right_set2.begin(), right_set2.end(), left_set2.begin(), left_set2.end(), inserter(common2, common2.begin()));
        common2 = left_set2;
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t2 - t1;
            cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;
        }
        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t3 - t2;
            cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        }
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].second, right_mask);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t5 - t4;
            cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        }
    }
    return index_list;
}

std::vector<pair<vector<int>, vector<int>>> coolB(vector<pair<vector<int>, vector<int>>>& index_list, std::vector<std::pair<int, int>> shape_list, std::vector<set<int>> right_nrows, int verbose = 0) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    vector<set<int>> col_idx(p);
    col_idx[0] = right_nrows[0];
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;
        // cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        set<int> common2;
        auto t1 = std::chrono::high_resolution_clock::now();
        set_intersection(right_nrows[i + 1].begin(), right_nrows[i + 1].end(), col_idx[i].begin(), col_idx[i].end(), inserter(common2, common2.begin()));
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t2 - t1;
            cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;
        }
        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t3 - t2;
            cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        }
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto remove_elements_set = [](vector<int>& vec, const vector<int>& mask, set<int>& col_idx) {
            vector<int> result;
            set<int> col_idx_set;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                    col_idx_set.insert(vec[i]);
                }
            }
            vec = result;
            col_idx = col_idx_set;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements_set(index_list[i + 1].second, right_mask, col_idx[i + 1]);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t5 - t4;
            cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        }
    }
    return index_list;
}

std::vector<pair<vector<int>, vector<int>>> coolBi(vector<pair<vector<int>, vector<int>>>& index_list, 
                                                    std::vector<std::pair<int, int>> shape_list, 
                                                    std::vector<set<int>> right_nrows, 
                                                    int verbose = 0) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    vector<set<int>> col_idx(p);
    col_idx[0] = right_nrows[0];
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;
        vector<int> tmp = index_list[i + 1].second; 
        // cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        set<int> common2;
        auto t1 = std::chrono::high_resolution_clock::now();
        set_intersection(right_nrows[i + 1].begin(), right_nrows[i + 1].end(), col_idx[i].begin(), col_idx[i].end(), inserter(common2, common2.begin()));
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t2 - t1;
            cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;
        }
        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t3 - t2;
            cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        }
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto remove_elements_set = [](vector<int>& vec, const vector<int>& mask, set<int>& col_idx) {
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    col_idx.insert(vec[i]);
                }
            }
            // col_idx = col_idx_set;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].second, right_mask);
            }
            #pragma omp section
            {
                remove_elements_set(tmp, right_mask, col_idx[i + 1]);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t5 - t4;
            cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        }
    }
    return index_list;
}

std::vector<pair<vector<int>, vector<int>>> coolfull(vector<pair<vector<int>, vector<int>>>& index_list, 
                                                    std::vector<std::pair<int, int>> shape_list, 
                                                    std::vector<set<int>> right_nrows, 
                                                    std::vector<set<int>> left_ncols,
                                                    int verbose = 0) {
    int p = index_list.size();
    vector<int> n_nodes;
    n_nodes.resize(p + 1, 0);
    vector<set<int>> col_idx(p);
    vector<set<int>> row_idx(p);
    col_idx[0] = right_nrows[0];
    auto fstart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < p - 1; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        vector<int>& left = index_list[i].second;
        vector<int>& right = index_list[i + 1].first;
        vector<int> tmp = index_list[i + 1].second; 
        vector<int> tmp2 = index_list[i].first; 

        // cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
        set<int> common2;
        auto t1 = std::chrono::high_resolution_clock::now();
        set_intersection(right_nrows[i + 1].begin(), right_nrows[i + 1].end(), col_idx[i].begin(), col_idx[i].end(), inserter(common2, common2.begin()));
        auto t2 = std::chrono::high_resolution_clock::now();
        left_ncols[i] = common2;
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t2 - t1;
            cout << "Time taken for set intersection: " << elapsed.count() << " seconds" << endl;
        }
        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        t2 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t3 - t2;
            cout << "Time taken for masking: " << elapsed.count() << " seconds" << endl;
        }
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto remove_elements_set = [](vector<int>& vec, const vector<int>& mask, set<int>& col_idx) {
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    col_idx.insert(vec[i]);
                }
            }
            // col_idx = col_idx_set;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i + 1].second, right_mask);
            }
            #pragma omp section
            {
                remove_elements_set(tmp, right_mask, col_idx[i + 1]);
            }
            #pragma omp section
            {
                remove_elements_set(tmp2, left_mask, row_idx[i]);
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        if (verbose>0) {
            std::chrono::duration<double> elapsed = t5 - t4;
            cout << "Time taken for removing elements: " << elapsed.count() << " seconds" << endl;
        }
    }
    auto fmid = std::chrono::high_resolution_clock::now();
    for (int i = p-2; i >= 1; --i) {
        vector<int>& right = index_list[i].first;
        vector<int>& left = index_list[i - 1].second;
        vector<int> tmp = index_list[i - 1].first;
        set<int> common2;
        set_intersection(row_idx[i].begin(), row_idx[i].end(), left_ncols[i - 1].begin(), left_ncols[i - 1].end(), inserter(common2, common2.begin()));
        
        vector<int> left_mask(left.size(), 0);
        vector<int> right_mask(right.size(), 0);
        int tnum = omp_get_max_threads();
        int chunk_size = ceil(1.0 * left.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int j = 0; j < left.size(); ++j) {
            if (common2.find(left[j]) != common2.end()) {
                left_mask[j] = 1;
            }
        }
        int right_chunk_size = ceil(1.0 * right.size() / (tnum * 8));
        #pragma omp parallel for schedule(dynamic, right_chunk_size)
        for (int j = 0; j < right.size(); ++j) {
            if (common2.find(right[j]) != common2.end()) {
                right_mask[j] = 1;
            }
        }
        auto remove_elements = [](vector<int>& vec, const vector<int>& mask) {
            vector<int> result;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    result.push_back(vec[i]);
                }
            }
            vec = result;
        };
        auto remove_elements_set = [](vector<int>& vec, const vector<int>& mask, set<int>& col_idx) {
            for (size_t i = 0; i < vec.size(); ++i) {
                if (mask[i]) {
                    col_idx.insert(vec[i]);
                }
            }
            // col_idx = col_idx_set;
        };
        auto t4 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                remove_elements(index_list[i].first, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i].second, right_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i - 1].first, left_mask);
            }
            #pragma omp section
            {
                remove_elements(index_list[i - 1].second, left_mask);
            }
            #pragma omp section
            {   
                row_idx[i - 1].clear();
                remove_elements_set(tmp, left_mask, row_idx[i - 1]);
            }
        }
    }
    auto fend = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for forward pass: " << std::chrono::duration<double>(fmid - fstart).count() << " seconds" << std::endl;
    std::cout << "Time taken for backward pass: " << std::chrono::duration<double>(fend - fmid).count() << " seconds" << std::endl;
    return index_list;
}