#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
// #include "IO.h"
#include "mkl.h"
#include "mkl_spblas.h"
#include <chrono>
#include "../utility.h"
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

using namespace Kokkos;
using namespace KokkosSparse;
typedef double Scalar;
typedef int Ordinal;
// typedef DefaultExecutionSpace DeviceType;
using DeviceType = Kokkos::OpenMP;
typedef CrsMatrix<Scalar, Ordinal, DeviceType, void, int> SparseMatrixType;
using namespace std;
size_t get_memory_usage() {
    std::ifstream file("/proc/self/status");
    std::string line;
    size_t memory_kb = 0;
    
    while (std::getline(file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label, unit;
            iss >> label >> memory_kb >> unit;
            break;
        }
    }
    
    return memory_kb * 1024; // 转换为字节
}

void coo2csr(
    const std::vector<int>& row_idx,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    int rows, int cols,
    std::vector<int>& csr_row_ptr,
    std::vector<int>& csr_col_idx,
    std::vector<double>& csr_values
) {
    int nnz = row_idx.size();
    csr_row_ptr.assign(rows + 1, 0);
    csr_col_idx.resize(nnz);
    csr_values.resize(nnz);

    for (int i = 0; i < nnz; ++i) {
        ++csr_row_ptr[row_idx[i] + 1];
    }
    // 2. 前缀和
    for (int i = 0; i < rows; ++i) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
    // 3. 填充 col_idx 和 values
    std::vector<int> offset = csr_row_ptr; // 记录每行已填充位置
    for (int i = 0; i < nnz; ++i) {
        int row = row_idx[i];
        int pos = offset[row]++;
        csr_col_idx[pos] = col_idx[i];
        csr_values[pos] = static_cast<double>(values[i]);
    }
}
// int smcm_mkl(std::vector<pair<vector<int>, vector<int>>> index_list,
//               std::vector<std::pair<int, int>> shape_list,
//               string order = "l2r",
//               int verbose = 0) {
//     int length = index_list.size();
//     std::vector<sparse_matrix_t> sparse_matrices(length);
//     std::vector<sparse_matrix_t> sparse_csr(length);
//     auto t0 = std::chrono::high_resolution_clock::now();
// #pragma omp parallel for
//     for (int i = 0; i < length; ++i) {
//         int nnz = std::get<0>(index_list[i]).size();
//         // 转换 float 到 double
//         std::vector<double> ones(nnz, 1.0);
//         mkl_sparse_d_create_coo(
//             &sparse_matrices[i],
//             SPARSE_INDEX_BASE_ZERO,
//             shape_list[i].first, shape_list[i].second,
//             nnz,
//             std::get<0>(index_list[i]).data(),
//             std::get<1>(index_list[i]).data(),
//             ones.data()
//         );
//         MKL_INT info;
//         info = mkl_sparse_convert_csr(sparse_matrices[i], SPARSE_OPERATION_NON_TRANSPOSE, &sparse_csr[i]);
//         if (info != 0) {
//             std::cerr << "Error converting to CSR format: " << info << std::endl;
//         }
//         mkl_sparse_destroy(sparse_matrices[i]);
//     }
//     auto t1 = std::chrono::high_resolution_clock::now();
//     if (verbose > 0) {
//         std::cout << "Time taken to create sparse matrices: "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
//                   << " seconds" << std::endl;
//     }
//     sparse_matrix_t result;
//     matrix_descr descL;
//     if (order == "l2r"){
//         mkl_sparse_copy(sparse_csr[0], descL, &result);
//         for (int i = 1; i < length; ++i) {
//             sparse_matrix_t temp_result;
//             MKL_INT status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, result, sparse_csr[i], &temp_result);
//             if (status != SPARSE_STATUS_SUCCESS) {
//                 std::cerr << "Matrix multiplication failed at step " << i << ", status: " << status << std::endl;
//                 break;
//             }
//             mkl_sparse_copy(temp_result, descL, &result);
//             mkl_sparse_destroy(temp_result);
//         }
//     } else{
//         return 0;
//     }
//     int nnz = get_nnz_from_csr(result);
//     auto t2 = std::chrono::high_resolution_clock::now();
//     if (verbose > 0) {
//         std::cout << "Final multiplied matrix non-zeros: " << nnz << std::endl;
//         std::cout << "Time taken for final multiplication: "
//                   << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
//                   << " seconds" << std::endl;
//     }
//     mkl_sparse_destroy(result);
//     return nnz;
// }
// void generate_parentheses(int start, int end, std::vector<std::string>& result) {
//     if (start == end) {
//         result.push_back("A" + std::to_string(start));
//         return;
//     }
//     for (int i = start; i < end; ++i) {
//         std::vector<std::string> left, right;
//         generate_parentheses(start, i, left);
//         generate_parentheses(i + 1, end, right);
//         for (const auto& l : left) {
//             for (const auto& r : right) {
//                 result.push_back("(" + l + " x " + r + ")");
//             }
//         }
//     }
// }

// 辅助函数：分割表达式
// std::pair<std::string, std::string> split_expression(const std::string& expr) {
//     int stack = 0;
//     for (size_t i = 0; i < expr.size(); ++i) {
//         if (expr[i] == '(') stack++;
//         else if (expr[i] == ')') stack--;
//         else if (expr[i] == 'x' && stack == 1) {
//             // 左右去掉外层括号
//             return {expr.substr(1, i - 2), expr.substr(i + 2, expr.size() - i - 3)};
//         }
//     }
//     return {"", ""};
// }
// int catalan_number(int n) {
//     int _n = n-1;
//     if (_n <= 1) return 1;
//     std::vector<long long> catalan(_n + 1, 0);
//     catalan[0] = catalan[1] = 1;
//     for (int i = 2; i <= _n; ++i) {
//         for (int j = 0; j < i; ++j) {
//             catalan[i] += catalan[j] * catalan[i - 1 - j];
//         }
//     }
//     return static_cast<int>(catalan[_n]);
// }


// sparse_matrix_t eval_parentheses(const std::string& expr, const std::vector<sparse_matrix_t>& sparse_matrices_csr) {
//     if (expr.find('x') == std::string::npos) {
//         int idx = std::stoi(expr.substr(1)) - 1;
//         return sparse_matrices_csr.at(idx);
//     }
//     auto [left, right] = split_expression(expr);
//     return eval_parentheses(left, sparse_matrices_csr) * eval_parentheses(right, sparse_matrices_csr); // 或用 matmul
// }
// void print_eval_steps(const std::string& expr, int depth = 0) {
//     // Indentation for better readability
//     std::string indent(depth * 2, ' ');
//     if (expr.find('x') == std::string::npos) {
//         std::cout << indent << "eval_parentheses(\"" << expr << "\") = " << expr << std::endl;
//         return;
//     }
//     std::cout << indent << "eval_parentheses(\"" << expr << "\")" << std::endl;
//     auto [left, right] = split_expression(expr);
//     print_eval_steps(left, depth + 1);
//     print_eval_steps(right, depth + 1);
//     std::cout << indent << "-> " << left << " * " << right << std::endl;
// }
// void smcm_get_order(const std::string& expr, std::vector<std::string>& order, int depth = 0) {
//     std::string indent(depth * 2, ' ');
//     if (expr.find('x') == std::string::npos) {
//         std::cout << indent << "eval_parentheses(\"" << expr << "\") = " << expr << std::endl;
//         return;
//     }
//     std::cout << indent << "eval_parentheses(\"" << expr << "\")" << std::endl;
//     auto [left, right] = split_expression(expr);
//     smcm_get_order(left, order, depth + 1);
//     smcm_get_order(right, order, depth + 1);
//     std::cout << indent << "-> " << left << " * " << right << std::endl;
//     std::string tmp = left + " x " + right;
//     order.push_back(tmp);
// }

int perform_smcm(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
         std::vector<std::pair<int, int>> shape_list,
         string order = "l2r",
         int verbose = 0) {
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<sparse_matrix_t> sparse_csr_matrices(length);
    std::vector<std::vector<int>> csr_row_ptrs(length);
    std::vector<std::vector<int>> csr_col_idxs(length);
    std::vector<std::vector<double>> csr_values(length);
    for (int i = 0; i < length; ++i) {
        const auto& row_idx = std::get<0>(coo_list[i]);
        const auto& col_idx = std::get<1>(coo_list[i]);
        const auto& vals    = std::get<2>(coo_list[i]);
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = row_idx.size();
        coo2csr(row_idx, col_idx, vals, rows, cols, csr_row_ptrs[i], csr_col_idxs[i], csr_values[i]);
        mkl_set_num_threads(8);
        MKL_INT info = mkl_sparse_d_create_csr(
                    &sparse_csr_matrices[i],
                    SPARSE_INDEX_BASE_ZERO,
                    shape_list[i].first, shape_list[i].second,
                    csr_row_ptrs[i].data(),
                    csr_row_ptrs[i].data() + 1,
                    csr_col_idxs[i].data(),
                    csr_values[i].data()
                );
        if (info != 0) {
            std::cerr << "Error creating CSR matrix: " << info << std::endl;
            continue;
        }
        // printf("Matrix %d created with shape (%d, %d) and nnz=%d\n", i, shape_list[i].first, shape_list[i].second, get_nnz_from_csr(sparse_csr_matrices[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    int result_nnz = 0;
    sparse_matrix_t result = sparse_csr_matrices[0];
    mkl_set_num_threads(16);
    if (order == "l2r") {
        for (int i = 1; i < length; ++i) {
            printf("Multiplying matrix %d\n", i);
            sparse_matrix_t temp_result;
            MKL_INT status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, result, sparse_csr_matrices[i], &temp_result);
            if (status != SPARSE_STATUS_SUCCESS) {
                std::cerr << "Matrix multiplication failed at step " << i << ", status: " << status << std::endl;
                break;
            }
            matrix_descr descL;
            mkl_sparse_copy(temp_result, descL, &result);
            mkl_sparse_destroy(temp_result);
        }
        result_nnz = get_nnz_from_csr(result);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        mkl_sparse_destroy(result);
        return result_nnz;
        
    } else{
        std::cerr << "Unsupported order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        std::map<std::string, sparse_matrix_t> matrix_map;
        
        return 0;
    }
}

int smcm_kk(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
         std::vector<std::pair<int, int>> shape_list,
         string order = "l2r",
         int verbose = 0) {
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    int result_nnz = 0;
    omp_set_num_threads(64);
    std::cout << "Kokkos 实际使用线程数: " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;
    using device = Kokkos::DefaultExecutionSpace;
    using memory_space = typename device::memory_space;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    std::vector<std::vector<int>> csr_row_ptrs(length);
    std::vector<std::vector<int>> csr_col_idxs(length);
    std::vector<std::vector<double>> csr_values(length);
    std::vector<crs_matrix_type> sparse_csr_matrices(length);
    for (int i = 0; i < length; ++i) {
        const auto& row_idx = std::get<0>(coo_list[i]);
        const auto& col_idx = std::get<1>(coo_list[i]);
        const auto& vals    = std::get<2>(coo_list[i]);
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = row_idx.size();
        coo2csr(row_idx, col_idx, vals, rows, cols, csr_row_ptrs[i], csr_col_idxs[i], csr_values[i]);
        View<int*, DeviceType> row_ptr("row_ptr", csr_row_ptrs[i].size());
        View<int*, DeviceType> col_indices("col_indices", csr_col_idxs[i].size());
        View<double*, DeviceType> values("values", csr_values[i].size());

        Kokkos::deep_copy(row_ptr, Kokkos::View<int*, Kokkos::HostSpace>(csr_row_ptrs[i].data(), csr_row_ptrs[i].size()));
        Kokkos::deep_copy(col_indices, Kokkos::View<int*, Kokkos::HostSpace>(csr_col_idxs[i].data(), csr_col_idxs[i].size()));
        Kokkos::deep_copy(values, Kokkos::View<double*, Kokkos::HostSpace>(csr_values[i].data(), csr_values[i].size()));

        // crs_matrix_type("A" + std::to_string(id), num_rows, num_cols, nnz, d_values, d_row_ptr, d_col_indices);
        sparse_csr_matrices[i] = crs_matrix_type("A" + std::to_string(i), rows, cols, nnz, values, row_ptr, col_indices);
    }
        
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    
    crs_matrix_type result = sparse_csr_matrices[0];
    if (order == "l2r") {
        for (int i = 1; i < length; ++i) {
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
            int, int, double, device, device, device>;
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_SERIAL);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            printf("Multiplying matrix %d\n", i);
            crs_matrix_type temp_result;
            KokkosSparse::spgemm_symbolic(
                kh, result, false, sparse_csr_matrices[i], false, temp_result
            );
            KokkosSparse::spgemm_numeric(
                kh, result, false, sparse_csr_matrices[i], false, temp_result
            );
            
            // kokkos_free(temp_result.graph.row_map);
            // kokkos_free(temp_result.graph.entries);
            // kokkos_free(temp_result.values);
            // kokkos_free(sparse_csr_matrices[i].graph.row_map);
            // kokkos_free(sparse_csr_matrices[i].graph.entries);
            // kokkos_free(sparse_csr_matrices[i].values);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto result_val = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.values);
        result_nnz = result_val.size();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        std::cout << "Result nnz: " << result_nnz << std::endl;
        result.values = decltype(result.values)();
        result.graph.row_map = decltype(result.graph.row_map)();
        result.graph.entries = decltype(result.graph.entries)();
    } else{
        return 0;
    }
    // Kokkos::finalize();
    return 0;
}
long multiply_by_order1(std::vector<std::string>& order, 
                       std::vector<sparse_matrix_t> &sparse_csr_matrices) {
    sparse_matrix_t temp_result;
    std::map<std::string, int> my_map;
    std::vector<sparse_matrix_t> intermediate_results(order.size());
    MKL_INT n_row, n_col;
    std::vector<int*> rows_start(order.size()), rows_end(order.size()), col_indices(order.size());
    std::vector<double*> csr_values(order.size());
    sparse_index_base_t csr_indexing;
    for (int i = 0; i < order.size(); ++i) {
        std::string o_prime = order[i];
        auto pos = o_prime.find('*');
        if (pos != std::string::npos) {
            o_prime.replace(pos, 1, "x");
        }
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        my_map[o_prime] = i;
    }
    for (int i = 0; i < order.size(); ++i ) {
        std::string o = order[i];
        auto pos = o.find('*');
        if (pos == std::string::npos) continue;
        std::string left = o.substr(0, pos);
        std::string right = o.substr(pos + 1);
        auto trim = [](const std::string& s) {
            auto start = s.find_first_not_of(" \t");
            auto end = s.find_last_not_of(" \t");
            return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
        };
        left = trim(left);
        right = trim(right);
        matrix_descr descL;
        sparse_matrix_t temp_result;
        if (left.front() == 'A' && left.find('(') == std::string::npos) {
            int left_idx = std::stoi(left.substr(1));
            // printf("left: %s, idx: %d\n", left.c_str(), left_idx-1);
            //left_mat = sparse_csr_matrices[idx-1];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                // printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], sparse_csr_matrices[right_idx-1], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            } else {
                int t = my_map[right];
                // printf("right: %s, corresponding my_map: %d\n", right.c_str(), t);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], intermediate_results[t], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                mkl_sparse_destroy(intermediate_results[t]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            }

        } else {
            // printf("left: %s\n", left.c_str());
            // left_mat = intermediate_results[my_map[left]];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                // printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], sparse_csr_matrices[right_idx-1], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            } else {
                // printf("right: %s\n", right.c_str());
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], intermediate_results[my_map[right]], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                mkl_sparse_destroy(intermediate_results[my_map[right]]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            }
        }
        std::string o_prime = o;
        o_prime.replace(pos, 1, "x");
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        // std::cout << "o_prime: " << o_prime << ", my_map[o_prime]: " << my_map[o_prime] << std::endl;
        intermediate_results[my_map[o_prime]] = temp_result;
    }
    long result_nnz = get_nnz_from_csr(intermediate_results[order.size()-1]);
    return result_nnz;
}
bool is_matrix_valid(sparse_matrix_t matrix) {
    // 调用 MKL 函数检查矩阵状态
    return matrix != nullptr; // 简化版
}

long multiply_by_order2(std::vector<std::string>& order, 
                       std::vector<sparse_matrix_t> &sparse_csr_matrices,
                       set<std::string>& failed_cases) {
    sparse_matrix_t temp_result;
    std::map<std::string, int> my_map;
    std::vector<sparse_matrix_t> intermediate_results(order.size());
    std::vector<bool> is_initialized(order.size(), false);
    MKL_INT n_row, n_col;
    std::vector<int*> rows_start(order.size()), rows_end(order.size()), col_indices(order.size());
    std::vector<double*> csr_values(order.size());
    sparse_index_base_t csr_indexing;
    for (int i = 0; i < order.size(); ++i) {
        std::cout << "order[" << i << "]: " << order[i] << std::endl;
        // std::cout << "order[" << i << "] find *: " << order[i].find('*') << std::endl;
    }
    for (int i = 0; i < order.size(); ++i) {
        std::string o_prime = order[i];
        if (failed_cases.count(order[i])) {
            std::cerr << "Par in failed case: " << order[i] << std::endl;
            return -1;
        }
        auto pos = o_prime.find('*');
        if (pos != std::string::npos) {
            o_prime.replace(pos, 1, "x");
        }
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        my_map[o_prime] = i;
    }
    for (int i = 0; i < order.size(); ++i ) {
        std::string o = order[i];
        cout << "Multiplying order[" << i << "]: " << o << std::endl;
        auto pos = o.find('*');
        if (pos == std::string::npos) continue;
        std::string left = o.substr(0, pos);
        std::string right = o.substr(pos + 1);
        auto trim = [](const std::string& s) {
            auto start = s.find_first_not_of(" \t");
            auto end = s.find_last_not_of(" \t");
            return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
        };
        left = trim(left);
        right = trim(right);
        
        matrix_descr descL;
        std::string o_prime = o;
        o_prime.replace(pos, 1, "x");
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        std::cout << "o_prime: " << o_prime << ", my_map[o_prime]: " << my_map[o_prime] << std::endl;
        if (left.front() == 'A' && left.find('(') == std::string::npos) {
            int left_idx = std::stoi(left.substr(1));
            printf("left: %s, idx: %d\n", left.c_str(), left_idx-1);
            //left_mat = sparse_csr_matrices[idx-1];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], sparse_csr_matrices[right_idx-1], &intermediate_results[my_map[o_prime]]
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    failed_cases.insert(o);
                    return -1;
                    break;
                } else {
                    is_initialized[my_map[o_prime]] = true;
                    std::cout << o_prime << my_map[o_prime] << " initialized: " << is_matrix_valid(intermediate_results[my_map[o_prime]]) << std::endl;
                }

                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            } else {
                
                auto it = my_map.find(right);
                if (it == my_map.end()) {
                    std::cerr << "Error: key not found for " << right << std::endl;
                    return -1;
                }
                int index = it->second;
                if (index >= i && !is_initialized[index]) {  // index >= i 表示还未计算
                    std::cerr << "Error: accessing uninitialized matrix for " << right << std::endl;
                    return -1;
                }
                int t = my_map[right];
                // printf("right: %s, corresponding my_map: %d\n", right.c_str(), t);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], intermediate_results[t], &intermediate_results[my_map[o_prime]]
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    failed_cases.insert(o);
                    return -1;
                }
                mkl_sparse_destroy(intermediate_results[t]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            }

        } else {
            // printf("left: %s\n", left.c_str());
            // left_mat = intermediate_results[my_map[left]];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                auto it = my_map.find(left);
                if (it == my_map.end() || !is_initialized[it->second]) {
                    std::cerr << "Error: accessing uninitialized matrix for " << left << std::endl;
                    return -1;
                }
                std::cout << "HERE+++++++++++++" << std::endl;
                std::cout << "Left Map" << my_map[left] << std::endl;
                std::cout << "Left Matrix" << is_matrix_valid(intermediate_results[my_map[left]]) << std::endl;
                std::cout << "Right Matrix" << is_matrix_valid(sparse_csr_matrices[right_idx-1]) << std::endl;
                std::cout << "Result Matrix" << is_matrix_valid(intermediate_results[my_map[o_prime]]) << std::endl;
                // printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], sparse_csr_matrices[right_idx-1], &intermediate_results[my_map[o_prime]]
                );

                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    failed_cases.insert(o);
                    return -1;
                }
                std::cout << "Multiply Finished" << std::endl;
                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            } else {
                auto it = my_map.find(right);
                if (it == my_map.end() || !is_initialized[it->second]) {
                    std::cerr << "Error: accessing uninitialized matrix for " << right << std::endl;
                    return -1;
                }
                it = my_map.find(left);
                if (it == my_map.end() || !is_initialized[it->second]) {
                    std::cerr << "Error: accessing uninitialized matrix for " << left << std::endl;
                    return -1;
                }
                // printf("right: %s\n", right.c_str());
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], intermediate_results[my_map[right]], &intermediate_results[my_map[o_prime]]
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    failed_cases.insert(o);
                    return -1;
                }
                mkl_sparse_destroy(intermediate_results[my_map[right]]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            }
        }
        std::cout << "Current my_map contents:" << std::endl;
        for (const auto& kv : my_map) {
            std::cout << "  \"" << kv.first << "\" : " << kv.second << std::endl;
        }
        // mkl_sparse_copy(temp_result, descL, &intermediate_results[my_map[o_prime]]);
        // intermediate_results[my_map[o_prime]] = temp_result;
    }
    long result_nnz = get_nnz_from_csr(intermediate_results[order.size()-1]);
    return result_nnz;
}
void print_memory_info() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0 || 
            line.find("SwapTotal:") == 0 || 
            line.find("SwapFree:") == 0) {
            std::cout << line << std::endl;
        }
    }
}
size_t get_available_memory() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t available_kb = 0;
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            std::istringstream iss(line);
            std::string label, unit;
            iss >> label >> available_kb >> unit;
            break;
        }
    }
    return available_kb * 1024;
}

long multiply_by_order3(std::vector<std::string>& order, 
                       std::vector<sparse_matrix_t> &sparse_csr_matrices,
                       std::set<std::string>& failed_cases) {
    
    std::map<std::string, sparse_matrix_t> intermediate_results;
    std::set<std::string> initialized_keys;
    
    // 初始化原始矩阵
    for (int i = 0; i < sparse_csr_matrices.size(); ++i) {
        std::string matrix_name = "A" + std::to_string(i + 1);
        intermediate_results[matrix_name] = sparse_csr_matrices[i];
        initialized_keys.insert(matrix_name);
    }
    
    auto trim = [](const std::string& s) {
        auto start = s.find_first_not_of(" \t");
        auto end = s.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    };
    
    // ✅ 修复1：更严格的内存管理
    size_t initial_memory = get_memory_usage();
    size_t MAX_TOTAL_MEMORY = 200LL * 1024 * 1024 * 1024; // 200GB 上限（2T的10%）
    size_t SINGLE_OP_THRESHOLD = 10LL * 1024 * 1024 * 1024; // 单次操作10GB阈值
    
    for (int i = 0; i < order.size(); ++i) {
        std::string o = order[i];
        std::cout << "Processing order[" << i << "]: " << o << std::endl;
        
        auto pos = o.find('*');
        if (pos == std::string::npos) continue;
        
        std::string left = trim(o.substr(0, pos));
        std::string right = trim(o.substr(pos + 1));
        std::string result_key = "(" + left + " x " + right + ")";
        
        // 检查操作数
        if (initialized_keys.find(left) == initialized_keys.end() || 
            initialized_keys.find(right) == initialized_keys.end()) {
            std::cerr << "Error: operand not found" << std::endl;
            return -1;
        }
        
        // ✅ 修复2：操作前内存检查
        size_t current_memory = get_memory_usage();
        if (current_memory > MAX_TOTAL_MEMORY) {
            std::cerr << "ERROR: Total memory usage exceeds " 
                     << MAX_TOTAL_MEMORY / (1024*1024*1024) << "GB limit!" << std::endl;
            return -1;
        }
        
        std::cout << "Current memory usage: " << current_memory / (1024*1024*1024) << "GB" << std::endl;
        
        // ✅ 修复3：立即清理策略
        // 在乘法前检查是否可以清理某些中间结果
        for (auto it = initialized_keys.begin(); it != initialized_keys.end();) {
            const std::string& key = *it;
            bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
            
            if (!is_original && key != left && key != right) {
                // 检查是否还会被后续操作使用
                bool needed_later = false;
                for (int j = i + 1; j < order.size(); ++j) {
                    if (order[j].find(key) != std::string::npos) {
                        needed_later = true;
                        break;
                    }
                }
                
                if (!needed_later) {
                    std::cout << "Cleaning up intermediate result: " << key << std::endl;
                    mkl_sparse_destroy(intermediate_results[key]);
                    intermediate_results.erase(key);
                    it = initialized_keys.erase(it);
                    continue;
                }
            }
            ++it;
        }
        size_t available = get_available_memory();
        std::cout << "Available memory: " << available / (1024*1024*1024) << "GB" << std::endl;
        if (available < 50LL * 1024 * 1024 * 1024) {  // 少于50GB可用
            std::cerr << "WARNING: Low memory, skipping operation" << std::endl;
            return -1;
        }
        size_t before_mem = get_memory_usage();
        std::cout << "Start Matmul" << std::endl;
        
        sparse_matrix_t temp_result;
        MKL_INT status = mkl_sparse_spmm(
            SPARSE_OPERATION_NON_TRANSPOSE, 
            intermediate_results[left], 
            intermediate_results[right], 
            &temp_result
        );
        
        size_t after_mem = get_memory_usage();
        size_t memory_growth = after_mem - before_mem;
        
        std::cout << "Memory growth for this operation: " 
                 << memory_growth / (1024*1024*1024) << "GB" << std::endl;
        
        // ✅ 修复4：严格的内存增长检查
        if (memory_growth > SINGLE_OP_THRESHOLD) {
            std::cerr << "ERROR: Single operation memory growth exceeds " 
                     << SINGLE_OP_THRESHOLD / (1024*1024*1024) << "GB!" << std::endl;
        }
        
        if (status != SPARSE_STATUS_SUCCESS) {
            std::cerr << "Matrix multiplication failed, status: " << status << std::endl;
            return -1;
        }
        
        std::cout << "Done Matmul" << std::endl;
        
        // 存储结果
        intermediate_results[result_key] = temp_result;
        initialized_keys.insert(result_key);
        
        // ✅ 修复4: 立即清理不再需要的操作数
        // 检查左操作数是否还会在后续使用
        if (left.front() != 'A' || left.find('(') != std::string::npos) {
            bool left_still_needed = false;
            for (int future_idx = i + 1; future_idx < order.size(); ++future_idx) {
                if (order[future_idx].find(left) != std::string::npos) {
                    left_still_needed = true;
                    break;
                }
            }
            if (!left_still_needed && intermediate_results.count(left)) {
                std::cout << "Immediate cleanup of left operand: " << left << std::endl;
                mkl_sparse_destroy(intermediate_results[left]);
                intermediate_results.erase(left);
                initialized_keys.erase(left);
            }
        }
        
        // 检查右操作数是否还会在后续使用
        if (right.front() != 'A' || right.find('(') != std::string::npos) {
            bool right_still_needed = false;
            for (int future_idx = i + 1; future_idx < order.size(); ++future_idx) {
                if (order[future_idx].find(right) != std::string::npos) {
                    right_still_needed = true;
                    break;
                }
            }
            if (!right_still_needed && intermediate_results.count(right)) {
                std::cout << "Immediate cleanup of right operand: " << right << std::endl;
                mkl_sparse_destroy(intermediate_results[right]);
                intermediate_results.erase(right);
                initialized_keys.erase(right);
            }
        }
        
        std::cout << "Successfully computed: " << result_key << std::endl;
    }
    
    // 获取最终结果并清理
    std::string final_expr = order.back();
    auto final_pos = final_expr.find('*');
    std::string final_left = trim(final_expr.substr(0, final_pos));
    std::string final_right = trim(final_expr.substr(final_pos + 1));
    std::string final_key = "(" + final_left + " x " + final_right + ")";
    
    long result_nnz = get_nnz_from_csr(intermediate_results[final_key]);
    
    // ✅ 修复5：改进的最终清理
    for (const auto& key : initialized_keys) {
        bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
        if (!is_original) {
            std::cout << "Final cleanup: " << key << std::endl;
            mkl_sparse_destroy(intermediate_results[key]);
        }
    }
    
    size_t final_memory = get_memory_usage();
    std::cout << "Memory usage after cleanup: " << final_memory / (1024*1024*1024) << "GB" << std::endl;
    std::cout << "Total memory growth: " << (final_memory - initial_memory) / (1024*1024*1024) << "GB" << std::endl;
    std::cout << "Result non-zeros: " << result_nnz << std::endl;
    return result_nnz;
}


long perform_smcm_coo(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
         std::vector<std::pair<int, int>> shape_list,
         string order = "l2r",
         int verbose = 0) {
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<sparse_matrix_t> sparse_csr_matrices(length), sparse_coo_matrices(length);
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = std::get<0>(coo_list[i]).size();
        std::vector<double> values_double(nnz, 1.0);
        MKL_INT info = mkl_sparse_d_create_coo(
                    &sparse_coo_matrices[i],
                    SPARSE_INDEX_BASE_ZERO,
                    shape_list[i].first, shape_list[i].second,
                    nnz,
                    std::get<0>(coo_list[i]).data(),
                    std::get<1>(coo_list[i]).data(),
                    values_double.data()
                );
        if (info != 0) {
            std::cerr << "Error creating COO matrix: " << info << std::endl;
            continue;
        }
        info = mkl_sparse_convert_csr(sparse_coo_matrices[i], SPARSE_OPERATION_NON_TRANSPOSE, &sparse_csr_matrices[i]);
        if (info != 0) {
            std::cerr << "Error converting to CSR format: " << info << std::endl;
        }
        mkl_sparse_destroy(sparse_coo_matrices[i]);
        // printf("Matrix %d created with shape (%d, %d) and nnz=%d\n", i, shape_list[i].first, shape_list[i].second, get_nnz_from_csr(sparse_csr_matrices[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    long result_nnz = 0;
    if (order == "l2r") {
        sparse_matrix_t result = sparse_csr_matrices[0];
        for (int i = 1; i < length; ++i) {
            printf("Multiplying matrix %d\n", i);
            sparse_matrix_t temp_result;
            MKL_INT status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, result, sparse_csr_matrices[i], &temp_result);
            if (status != SPARSE_STATUS_SUCCESS) {
                std::cerr << "Matrix multiplication failed at step " << i << ", status: " << status << std::endl;
                break;
            }
            matrix_descr descL;
            mkl_sparse_copy(temp_result, descL, &result);
            mkl_sparse_destroy(temp_result);
        }
        result_nnz = get_nnz_from_csr(result);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        mkl_sparse_destroy(result);
        return result_nnz;
        
    } else{
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        result_nnz = multiply_by_order1(order_vec, sparse_csr_matrices);
        
        return result_nnz;
    }
}

long smcm_coo(std::vector<sparse_matrix_t> sparse_csr_matrices,
         std::vector<std::pair<int, int>> shape_list,
         std::set<std::string>& failed_cases,
         string order = "l2r",
         int verbose = 0) {

    int length = shape_list.size();
    auto t1 = std::chrono::high_resolution_clock::now();
    long result_nnz = 0;
    ///
    if (order == "l2r") {
        sparse_matrix_t result = sparse_csr_matrices[0];
        for (int i = 1; i < length; ++i) {
            printf("Multiplying matrix %d\n", i);
            sparse_matrix_t temp_result;
            MKL_INT status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, result, sparse_csr_matrices[i], &temp_result);
            if (status != SPARSE_STATUS_SUCCESS) {
                std::cerr << "Matrix multiplication failed at step " << i << ", status: " << status << std::endl;
                break;
            }
            matrix_descr descL;
            mkl_sparse_copy(temp_result, descL, &result);
            mkl_sparse_destroy(temp_result);
        }
        result_nnz = get_nnz_from_csr(result);
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        mkl_sparse_destroy(result);
        return result_nnz;
        
    } else{
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        result_nnz = multiply_by_order3(order_vec, sparse_csr_matrices, failed_cases);

        return result_nnz;
    }
}

std::pair<long, long> multiply_by_order_mul(std::vector<std::string>& order, 
                       std::vector<sparse_matrix_t> &sparse_csr_matrices) {
    sparse_matrix_t temp_result;
    std::map<std::string, int> my_map;
    std::vector<sparse_matrix_t> intermediate_results(order.size());
    MKL_INT n_row, n_col;
    std::vector<int*> rows_start(order.size()), rows_end(order.size()), col_indices(order.size());
    std::vector<double*> csr_values(order.size());
    sparse_index_base_t csr_indexing;
    for (int i = 0; i < order.size(); ++i) {
        std::string o_prime = order[i];
        auto pos = o_prime.find('*');
        if (pos != std::string::npos) {
            o_prime.replace(pos, 1, "x");
        }
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        my_map[o_prime] = i;
    }
    long sum = 0;
    long inter_nnz;
    for (int i = 0; i < order.size(); ++i ) {
        std::string o = order[i];
        auto pos = o.find('*');
        if (pos == std::string::npos) continue;
        std::string left = o.substr(0, pos);
        std::string right = o.substr(pos + 1);
        auto trim = [](const std::string& s) {
            auto start = s.find_first_not_of(" \t");
            auto end = s.find_last_not_of(" \t");
            return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
        };
        left = trim(left);
        right = trim(right);
        matrix_descr descL;
        sparse_matrix_t temp_result;
        if (left.front() == 'A' && left.find('(') == std::string::npos) {
            int left_idx = std::stoi(left.substr(1));
            // printf("left: %s, idx: %d\n", left.c_str(), left_idx-1);
            //left_mat = sparse_csr_matrices[idx-1];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                // printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], sparse_csr_matrices[right_idx-1], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            } else {
                int t = my_map[right];
                // printf("right: %s, corresponding my_map: %d\n", right.c_str(), t);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, sparse_csr_matrices[left_idx-1], intermediate_results[t], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                mkl_sparse_destroy(intermediate_results[t]);
                // mkl_sparse_destroy(sparse_csr_matrices[left_idx-1]);
            }

        } else {
            // printf("left: %s\n", left.c_str());
            // left_mat = intermediate_results[my_map[left]];
            if (right.front() == 'A' && right.find('(') == std::string::npos) {
                int right_idx = std::stoi(right.substr(1));
                // printf("right: %s, idx: %d\n", right.c_str(), right_idx-1);
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], sparse_csr_matrices[right_idx-1], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                // mkl_sparse_destroy(sparse_csr_matrices[right_idx-1]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            } else {
                // printf("right: %s\n", right.c_str());
                MKL_INT status = mkl_sparse_spmm(
                    SPARSE_OPERATION_NON_TRANSPOSE, intermediate_results[my_map[left]], intermediate_results[my_map[right]], &temp_result
                );
                if (status != SPARSE_STATUS_SUCCESS) {
                    std::cerr << "Matrix multiplication failed for " << o << ", status: " << status << std::endl;
                    break;
                }
                mkl_sparse_destroy(intermediate_results[my_map[right]]);
                mkl_sparse_destroy(intermediate_results[my_map[left]]);
            }
        }
        std::string o_prime = o;
        o_prime.replace(pos, 1, "x");
        o_prime = o_prime + ")";
        o_prime = "(" + o_prime;
        sparse_matrix_t save_result;
        
        sparse_index_base_t indexing;
        MKL_INT rows, cols;
        const MKL_INT *row_ptr, *row_end, *col_ind;
        const double *values;
        MKL_INT nnz;

        // 导出CSR格式数据
        MKL_INT export_status = mkl_sparse_d_export_csr(
            temp_result, &indexing, &rows, &cols, 
            const_cast<MKL_INT**>(&row_ptr), 
            const_cast<MKL_INT**>(&row_end), 
            const_cast<MKL_INT**>(&col_ind), 
            const_cast<double**>(&values)
        );
        
        if (export_status != SPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to export CSR matrix, status: " << export_status << std::endl;
            mkl_sparse_destroy(temp_result);
            continue;
        }
        std::cout << "Failed Here 0" << std::endl;
        // 计算非零元素个数
        nnz = row_ptr[rows] - row_ptr[0];
        if (i == 0) {
            inter_nnz = nnz;
        }

        // 计算所有值的和
        #pragma omp parallel for reduction(+:sum)
        for (MKL_INT j = 0; j < nnz; ++j) {
            sum += values[j];
        }
        std::cout << "Failed Here" << std::endl;
        // ✅ 创建数据副本以避免 const 问题
        std::vector<MKL_INT> row_ptr_copy(row_ptr, row_ptr + rows + 1);
        std::vector<MKL_INT> col_ind_copy(col_ind, col_ind + nnz);
        std::vector<double> ones(nnz, 1.0);
        
        // 使用副本创建新矩阵
        MKL_INT create_status = mkl_sparse_d_create_csr(
            &save_result, indexing, rows, cols, 
            row_ptr_copy.data(), 
            row_ptr_copy.data() + 1, 
            col_ind_copy.data(), 
            ones.data()
        );
        std::cout << "Failed Here 1" << std::endl;
        if (create_status != SPARSE_STATUS_SUCCESS) {
            std::cerr << "Failed to create CSR matrix, status: " << create_status << std::endl;
            mkl_sparse_destroy(temp_result);
            continue;
        }
        
        intermediate_results[my_map[o_prime]] = save_result;
    }
    // long result_nnz = get_nnz_from_csr(intermediate_results[order.size()-1]);
    return std::make_pair(sum, inter_nnz);
}

std::pair<long, long> perform_smcm_coo_mul(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
         std::vector<std::pair<int, int>> shape_list,
         string order = "l2r",
         int verbose = 0) {
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<sparse_matrix_t> sparse_csr_matrices(length), sparse_coo_matrices(length);
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = std::get<0>(coo_list[i]).size();
        std::vector<double> values_double(nnz, 1.0);
        MKL_INT info = mkl_sparse_d_create_coo(
                    &sparse_coo_matrices[i],
                    SPARSE_INDEX_BASE_ZERO,
                    shape_list[i].first, shape_list[i].second,
                    nnz,
                    std::get<0>(coo_list[i]).data(),
                    std::get<1>(coo_list[i]).data(),
                    values_double.data()
                );
        if (info != 0) {
            std::cerr << "Error creating COO matrix: " << info << std::endl;
            continue;
        }
        info = mkl_sparse_convert_csr(sparse_coo_matrices[i], SPARSE_OPERATION_NON_TRANSPOSE, &sparse_csr_matrices[i]);
        if (info != 0) {
            std::cerr << "Error converting to CSR format: " << info << std::endl;
        }
        mkl_sparse_destroy(sparse_coo_matrices[i]);
        // printf("Matrix %d created with shape (%d, %d) and nnz=%d\n", i, shape_list[i].first, shape_list[i].second, get_nnz_from_csr(sparse_csr_matrices[i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    long n_mul, inter_nnz;
    if (order == "l2r") {
        // sparse_matrix_t result = sparse_csr_matrices[0];
        // for (int i = 1; i < length; ++i) {
        //     printf("Multiplying matrix %d\n", i);
        //     sparse_matrix_t temp_result;
        //     MKL_INT status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, result, sparse_csr_matrices[i], &temp_result);
        //     if (status != SPARSE_STATUS_SUCCESS) {
        //         std::cerr << "Matrix multiplication failed at step " << i << ", status: " << status << std::endl;
        //         break;
        //     }
        //     matrix_descr descL;
        //     mkl_sparse_copy(temp_result, descL, &result);
        //     mkl_sparse_destroy(temp_result);
        // }
        // auto t2 = std::chrono::high_resolution_clock::now();
        // if (verbose > 0) {
        //     std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
        //     std::cout << "Time taken for final multiplication: "
        //             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
        //             << " seconds" << std::endl;
        // }
        // mkl_sparse_destroy(result);
        // return result_nnz;
        return std::make_pair(0, 0);
        
    } else{
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        std::tie(n_mul, inter_nnz) = multiply_by_order_mul(order_vec, sparse_csr_matrices);

        return std::make_pair(n_mul, inter_nnz);
    }
}