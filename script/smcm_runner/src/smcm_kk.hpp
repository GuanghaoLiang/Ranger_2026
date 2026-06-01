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


template<typename IndexType, typename ValueType>
void coo2csr_new(const std::vector<IndexType>& row_idx,
             const std::vector<IndexType>& col_idx, 
             const std::vector<ValueType>& vals,
             IndexType num_rows,
             IndexType num_cols,
             std::vector<IndexType>& csr_row_ptrs,
             std::vector<IndexType>& csr_col_idxs,
             std::vector<double>& csr_values) {
    
    if (row_idx.size() != col_idx.size() || row_idx.size() != vals.size()) {
        throw std::invalid_argument("COO arrays must have the same size");
    }
    
    size_t nnz = row_idx.size();
    
    // 初始化输出数组
    csr_row_ptrs.assign(num_rows + 1, 0);
    csr_col_idxs.resize(nnz);
    csr_values.resize(nnz);
    
    if (nnz == 0) {
        return;
    }
    
    // 步骤1: 计算每行的非零元个数
    for (size_t i = 0; i < nnz; ++i) {
        if (row_idx[i] < 0 || row_idx[i] >= num_rows) {
            throw std::out_of_range("Row index out of range: " + std::to_string(row_idx[i]));
        }
        if (col_idx[i] < 0 || col_idx[i] >= num_cols) {
            throw std::out_of_range("Column index out of range: " + std::to_string(col_idx[i]));
        }
        csr_row_ptrs[row_idx[i] + 1]++;
    }
    
    // 步骤2: 计算累积和得到行指针
    for (IndexType i = 1; i <= num_rows; ++i) {
        csr_row_ptrs[i] += csr_row_ptrs[i - 1];
    }
    
    // 步骤3: 创建临时数组来跟踪当前行的插入位置
    std::vector<IndexType> row_counters(csr_row_ptrs.begin(), csr_row_ptrs.end() - 1);
    
    // 步骤4: 填充列索引和值数组
    for (size_t i = 0; i < nnz; ++i) {
        IndexType row = row_idx[i];
        IndexType pos = row_counters[row]++;
        
        csr_col_idxs[pos] = col_idx[i];
        csr_values[pos] = static_cast<double>(vals[i]);
    }
    
    // 步骤5: 对每行的列索引进行排序（可选，但推荐）
    for (IndexType row = 0; row < num_rows; ++row) {
        IndexType start = csr_row_ptrs[row];
        IndexType end = csr_row_ptrs[row + 1];
        
        if (end > start) {
            // 创建索引-值对进行排序
            std::vector<std::pair<IndexType, double>> pairs;
            pairs.reserve(end - start);
            
            for (IndexType j = start; j < end; ++j) {
                pairs.emplace_back(csr_col_idxs[j], csr_values[j]);
            }
            
            // 按列索引排序
            std::sort(pairs.begin(), pairs.end());
            
            // 写回排序后的结果
            for (size_t j = 0; j < pairs.size(); ++j) {
                csr_col_idxs[start + j] = pairs[j].first;
                csr_values[start + j] = pairs[j].second;
            }
        }
    }
}
// Kokkos 版本的矩阵链乘函数
long multiply_by_order_kokkos(std::vector<std::string>& order, 
                             std::vector<KokkosSparse::CrsMatrix<double, int, Kokkos::DefaultExecutionSpace, void, int>>& sparse_csr_matrices,
                             std::set<std::string>& failed_cases) {
    
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, device, device, device>;
    
    // std::cout << "\n=== Starting multiply_by_order_kokkos ===" << std::endl;
    
    std::map<std::string, crs_matrix_type> intermediate_results;
    std::set<std::string> initialized_keys;
    
    // 初始化原始矩阵
    for (int i = 0; i < sparse_csr_matrices.size(); ++i) {
        std::string matrix_name = "A" + std::to_string(i + 1);
        intermediate_results[matrix_name] = sparse_csr_matrices[i];
        initialized_keys.insert(matrix_name);
        // std::cout << "Initialized matrix: " << matrix_name << std::endl;
    }
    
    auto trim = [](const std::string& s) {
        auto start = s.find_first_not_of(" \t");
        auto end = s.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    };
    
    // 预分析引用计数
    std::map<std::string, int> ref_count;
    std::map<std::string, std::vector<int>> usage_steps;
    
    // std::cout << "\n=== Analyzing reference counts ===" << std::endl;
    for (int i = 0; i < order.size(); ++i) {
        const auto& expr = order[i];
        auto pos = expr.find('*');
        if (pos != std::string::npos) {
            std::string left = trim(expr.substr(0, pos));
            std::string right = trim(expr.substr(pos + 1));
            
            if (left.front() != 'A' || left.find('(') != std::string::npos) {
                usage_steps[left].push_back(i);
                ref_count[left]++;
            }
            if (right.front() != 'A' || right.find('(') != std::string::npos) {
                usage_steps[right].push_back(i);
                ref_count[right]++;
            }
        }
    }
    
    // 打印引用分析
    // for (const auto& pair : ref_count) {
    //     std::cout << "  \"" << pair.first << "\": used " << pair.second << " times" << std::endl;
    // }
    
    // std::cout << "\n=== Starting matrix multiplication sequence ===" << std::endl;
    
    for (int i = 0; i < order.size(); ++i) {
        std::string o = order[i];
        // std::cout << "\n--- Processing order[" << i << "]: " << o << " ---" << std::endl;
        
        auto pos = o.find('*');
        if (pos == std::string::npos) {
            std::cout << "Skipping invalid expression: " << o << std::endl;
            continue;
        }
        
        std::string left = trim(o.substr(0, pos));
        std::string right = trim(o.substr(pos + 1));
        std::string result_key = "(" + left + " x " + right + ")";
        
        // std::cout << "Left operand: \"" << left << "\"" << std::endl;
        // std::cout << "Right operand: \"" << right << "\"" << std::endl;
        // std::cout << "Result key: \"" << result_key << "\"" << std::endl;
        
        // 检查操作数是否存在
        if (initialized_keys.find(left) == initialized_keys.end()) {
            std::cerr << "ERROR: Left operand not found: " << left << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        if (initialized_keys.find(right) == initialized_keys.end()) {
            std::cerr << "ERROR: Right operand not found: " << right << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        // 执行 Kokkos 矩阵乘法
        // std::cout << "Starting Kokkos matrix multiplication..." << std::endl;
        
        try {
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_KK);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            crs_matrix_type temp_result;
            
            // 符号阶段
            KokkosSparse::spgemm_symbolic(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // 数值阶段
            KokkosSparse::spgemm_numeric(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // std::cout << "Kokkos matrix multiplication completed successfully" << std::endl;
            
            // 存储结果
            intermediate_results[result_key] = temp_result;
            initialized_keys.insert(result_key);
            
            // 清理 kernel handle
            kh.destroy_spgemm_handle();
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Kokkos matrix multiplication failed: " << e.what() << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        // 基于引用计数的清理
        auto safe_cleanup = [&](const std::string& key) {
            if (key.front() != 'A' || key.find('(') != std::string::npos) {
                if (ref_count.count(key) && ref_count[key] > 0) {
                    ref_count[key]--;
                    // std::cout << "Decremented ref count for \"" << key 
                            //  << "\" to " << ref_count[key] << std::endl;
                    
                    if (ref_count[key] == 0) {
                        // std::cout << "Cleaning up: \"" << key << "\" (ref count reached 0)" << std::endl;
                        if (intermediate_results.count(key)) {
                            intermediate_results.erase(key);
                            initialized_keys.erase(key);
                        }
                    }
                }
            }
        };
        
        safe_cleanup(left);
        safe_cleanup(right);
        
        // std::cout << "Successfully computed: \"" << result_key << "\"" << std::endl;
        // std::cout << "Active matrices: " << initialized_keys.size() << std::endl;
        
        // 周期性清理检查
        if (i % 3 == 2) {
            // std::cout << "Performing periodic cleanup check..." << std::endl;
            
            for (auto it = intermediate_results.begin(); it != intermediate_results.end();) {
                const std::string& key = it->first;
                bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
                
                if (!is_original && key != result_key) {
                    bool needed_soon = false;
                    for (int j = i + 1; j < std::min(i + 3, (int)order.size()); ++j) {
                        if (order[j].find(key) != std::string::npos) {
                            needed_soon = true;
                            break;
                        }
                    }
                    
                    if (!needed_soon) {
                        // std::cout << "Force cleaning: \"" << key << "\"" << std::endl;
                        initialized_keys.erase(key);
                        if (ref_count.count(key)) {
                            ref_count[key] = 0;
                        }
                        it = intermediate_results.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
    }
    
    // std::cout << "\n=== Finalizing results ===" << std::endl;
    
    // 获取最终结果
    if (order.empty()) {
        std::cerr << "ERROR: Empty order vector" << std::endl;
        return -1;
    }
    
    std::string final_expr = order.back();
    auto final_pos = final_expr.find('*');
    if (final_pos == std::string::npos) {
        std::cerr << "ERROR: Invalid final expression: " << final_expr << std::endl;
        return -1;
    }
    
    std::string final_left = trim(final_expr.substr(0, final_pos));
    std::string final_right = trim(final_expr.substr(final_pos + 1));
    std::string final_key = "(" + final_left + " x " + final_right + ")";
    
    // std::cout << "Final result key: \"" << final_key << "\"" << std::endl;
    
    if (initialized_keys.find(final_key) == initialized_keys.end()) {
        std::cerr << "ERROR: Final result not found: " << final_key << std::endl;
        return -1;
    }
    
    // 获取最终结果的非零元数量
    auto final_matrix = intermediate_results[final_key];
    auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_matrix.values);
    long result_nnz = result_values.extent(0);
    
    std::cout << "Final result nnz: " << result_nnz << std::endl;
    
    // 最终清理所有中间结果
    // std::cout << "\n=== Final cleanup ===" << std::endl;
    for (const auto& key : initialized_keys) {
        bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
        if (!is_original) {
            std::cout << "Final cleanup: \"" << key << "\"" << std::endl;
        }
    }
    
    std::cout << "=== multiply_by_order_kokkos completed successfully ===" << std::endl;
    return result_nnz;
}
long multiply_by_order_kokkos_seq(std::vector<std::string>& order, 
                             std::vector<KokkosSparse::CrsMatrix<double, int, Kokkos::DefaultExecutionSpace, void, int>>& sparse_csr_matrices,
                             std::set<std::string>& failed_cases) {
    
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, device, device, device>;
    
    // std::cout << "\n=== Starting multiply_by_order_kokkos ===" << std::endl;
    
    std::map<std::string, crs_matrix_type> intermediate_results;
    std::set<std::string> initialized_keys;
    
    // 初始化原始矩阵
    for (int i = 0; i < sparse_csr_matrices.size(); ++i) {
        std::string matrix_name = "A" + std::to_string(i + 1);
        intermediate_results[matrix_name] = sparse_csr_matrices[i];
        initialized_keys.insert(matrix_name);
        // std::cout << "Initialized matrix: " << matrix_name << std::endl;
    }
    
    auto trim = [](const std::string& s) {
        auto start = s.find_first_not_of(" \t");
        auto end = s.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    };
    
    // 预分析引用计数
    std::map<std::string, int> ref_count;
    std::map<std::string, std::vector<int>> usage_steps;
    
    // std::cout << "\n=== Analyzing reference counts ===" << std::endl;
    for (int i = 0; i < order.size(); ++i) {
        const auto& expr = order[i];
        auto pos = expr.find('*');
        if (pos != std::string::npos) {
            std::string left = trim(expr.substr(0, pos));
            std::string right = trim(expr.substr(pos + 1));
            
            if (left.front() != 'A' || left.find('(') != std::string::npos) {
                usage_steps[left].push_back(i);
                ref_count[left]++;
            }
            if (right.front() != 'A' || right.find('(') != std::string::npos) {
                usage_steps[right].push_back(i);
                ref_count[right]++;
            }
        }
    }
    
    // 打印引用分析
    // for (const auto& pair : ref_count) {
    //     std::cout << "  \"" << pair.first << "\": used " << pair.second << " times" << std::endl;
    // }
    
    // std::cout << "\n=== Starting matrix multiplication sequence ===" << std::endl;
    
    for (int i = 0; i < order.size(); ++i) {
        std::string o = order[i];
        // std::cout << "\n--- Processing order[" << i << "]: " << o << " ---" << std::endl;
        
        auto pos = o.find('*');
        if (pos == std::string::npos) {
            std::cout << "Skipping invalid expression: " << o << std::endl;
            continue;
        }
        
        std::string left = trim(o.substr(0, pos));
        std::string right = trim(o.substr(pos + 1));
        std::string result_key = "(" + left + " x " + right + ")";
        
        // std::cout << "Left operand: \"" << left << "\"" << std::endl;
        // std::cout << "Right operand: \"" << right << "\"" << std::endl;
        // std::cout << "Result key: \"" << result_key << "\"" << std::endl;
        
        // 检查操作数是否存在
        if (initialized_keys.find(left) == initialized_keys.end()) {
            std::cerr << "ERROR: Left operand not found: " << left << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        if (initialized_keys.find(right) == initialized_keys.end()) {
            std::cerr << "ERROR: Right operand not found: " << right << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        // 执行 Kokkos 矩阵乘法
        // std::cout << "Starting Kokkos matrix multiplication..." << std::endl;
        
        try {
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_DEFAULT);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            crs_matrix_type temp_result;
            
            // 符号阶段
            KokkosSparse::spgemm_symbolic(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // 数值阶段
            KokkosSparse::spgemm_numeric(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // std::cout << "Kokkos matrix multiplication completed successfully" << std::endl;
            
            // 存储结果
            intermediate_results[result_key] = temp_result;
            initialized_keys.insert(result_key);
            
            // 清理 kernel handle
            kh.destroy_spgemm_handle();
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Kokkos matrix multiplication failed: " << e.what() << std::endl;
            failed_cases.insert(o);
            return -1;
        }
        
        // 基于引用计数的清理
        auto safe_cleanup = [&](const std::string& key) {
            if (key.front() != 'A' || key.find('(') != std::string::npos) {
                if (ref_count.count(key) && ref_count[key] > 0) {
                    ref_count[key]--;
                    // std::cout << "Decremented ref count for \"" << key 
                            //  << "\" to " << ref_count[key] << std::endl;
                    
                    if (ref_count[key] == 0) {
                        // std::cout << "Cleaning up: \"" << key << "\" (ref count reached 0)" << std::endl;
                        if (intermediate_results.count(key)) {
                            intermediate_results.erase(key);
                            initialized_keys.erase(key);
                        }
                    }
                }
            }
        };
        
        safe_cleanup(left);
        safe_cleanup(right);
        
        // std::cout << "Successfully computed: \"" << result_key << "\"" << std::endl;
        // std::cout << "Active matrices: " << initialized_keys.size() << std::endl;
        
        // 周期性清理检查
        if (i % 3 == 2) {
            // std::cout << "Performing periodic cleanup check..." << std::endl;
            
            for (auto it = intermediate_results.begin(); it != intermediate_results.end();) {
                const std::string& key = it->first;
                bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
                
                if (!is_original && key != result_key) {
                    bool needed_soon = false;
                    for (int j = i + 1; j < std::min(i + 3, (int)order.size()); ++j) {
                        if (order[j].find(key) != std::string::npos) {
                            needed_soon = true;
                            break;
                        }
                    }
                    
                    if (!needed_soon) {
                        // std::cout << "Force cleaning: \"" << key << "\"" << std::endl;
                        initialized_keys.erase(key);
                        if (ref_count.count(key)) {
                            ref_count[key] = 0;
                        }
                        it = intermediate_results.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
    }
    
    // std::cout << "\n=== Finalizing results ===" << std::endl;
    
    // 获取最终结果
    if (order.empty()) {
        std::cerr << "ERROR: Empty order vector" << std::endl;
        return -1;
    }
    
    std::string final_expr = order.back();
    auto final_pos = final_expr.find('*');
    if (final_pos == std::string::npos) {
        std::cerr << "ERROR: Invalid final expression: " << final_expr << std::endl;
        return -1;
    }
    
    std::string final_left = trim(final_expr.substr(0, final_pos));
    std::string final_right = trim(final_expr.substr(final_pos + 1));
    std::string final_key = "(" + final_left + " x " + final_right + ")";
    
    // std::cout << "Final result key: \"" << final_key << "\"" << std::endl;
    
    if (initialized_keys.find(final_key) == initialized_keys.end()) {
        std::cerr << "ERROR: Final result not found: " << final_key << std::endl;
        return -1;
    }
    
    // 获取最终结果的非零元数量
    auto final_matrix = intermediate_results[final_key];
    auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_matrix.values);
    long result_nnz = result_values.extent(0);
    
    std::cout << "Final result nnz: " << result_nnz << std::endl;
    
    // 最终清理所有中间结果
    // std::cout << "\n=== Final cleanup ===" << std::endl;
    for (const auto& key : initialized_keys) {
        bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
        if (!is_original) {
            std::cout << "Final cleanup: \"" << key << "\"" << std::endl;
        }
    }
    
    std::cout << "=== multiply_by_order_kokkos completed successfully ===" << std::endl;
    return result_nnz;
}

long smcm_coo_kokkos(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
                    std::vector<std::pair<int, int>> shape_list,
                    std::set<std::string>& failed_cases,
                    string order = "l2r",
                    int verbose = 0) {
    if (!Kokkos::is_initialized()) {
        std::cerr << "ERROR: Kokkos is not initialized. Please call Kokkos::initialize() in main()." << std::endl;
        return -1;
    }
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    long result_nnz = 0;
    std::cout << "============Kokkos number of threads: " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;
    
    std::vector<std::vector<int>> csr_row_ptrs(length);
    std::vector<std::vector<int>> csr_col_idxs(length);
    std::vector<std::vector<double>> csr_values(length);
    std::vector<crs_matrix_type> sparse_csr_matrices(length);
    
    // 数据转换和矩阵创建
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        const auto& row_idx = std::get<0>(coo_list[i]);
        const auto& col_idx = std::get<1>(coo_list[i]);
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = row_idx.size();
        // Cast vals to double
        std::vector<double> vals(row_idx.size());
        const auto& vals_src = std::get<2>(coo_list[i]);
        for (size_t j = 0; j < vals_src.size(); ++j) {
            vals[j] = static_cast<double>(vals_src[j]);
        }

        coo2csr_new<int, double>(row_idx, col_idx, vals, rows, cols, csr_row_ptrs[i], csr_col_idxs[i], csr_values[i]);

        View<int*, DeviceType> row_ptr("row_ptr", csr_row_ptrs[i].size());
        View<int*, DeviceType> col_indices("col_indices", csr_col_idxs[i].size());
        View<double*, DeviceType> values("values", csr_values[i].size());

        Kokkos::deep_copy(row_ptr, Kokkos::View<int*, Kokkos::HostSpace>(csr_row_ptrs[i].data(), csr_row_ptrs[i].size()));
        Kokkos::deep_copy(col_indices, Kokkos::View<int*, Kokkos::HostSpace>(csr_col_idxs[i].data(), csr_col_idxs[i].size()));
        Kokkos::deep_copy(values, Kokkos::View<double*, Kokkos::HostSpace>(csr_values[i].data(), csr_values[i].size()));

        sparse_csr_matrices[i] = crs_matrix_type("A" + std::to_string(i), rows, cols, nnz, values, row_ptr, col_indices);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    
    if (order == "l2r") {
        // 简单的左到右乘法
        crs_matrix_type result = sparse_csr_matrices[0];
        
        for (int i = 1; i < length; ++i) {
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                int, int, double, device, device, device>;
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_KK);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            // printf("Multiplying matrix %d\n", i);
            crs_matrix_type temp_result;
            
            try {
                KokkosSparse::spgemm_symbolic(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                KokkosSparse::spgemm_numeric(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                
                result = temp_result;
                
            } catch (const std::exception& e) {
                std::cerr << "Matrix multiplication failed at step " << i << ": " << e.what() << std::endl;
                failed_cases.insert("step_" + std::to_string(i));
                kh.destroy_spgemm_handle();
                return -1;
            }
            
            kh.destroy_spgemm_handle();
        }
        
        auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.values);
        result_nnz = result_values.extent(0);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        
        return result_nnz;
        
    } else {
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        
        result_nnz = multiply_by_order_kokkos(order_vec, sparse_csr_matrices, failed_cases);
        
        return result_nnz;
    }
}
long smcm_coo_kokkos_seq(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
                    std::vector<std::pair<int, int>> shape_list,
                    std::set<std::string>& failed_cases,
                    string order = "l2r",
                    int verbose = 0) {
    if (!Kokkos::is_initialized()) {
        std::cerr << "ERROR: Kokkos is not initialized. Please call Kokkos::initialize() in main()." << std::endl;
        return -1;
    }
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    long result_nnz = 0;
    std::cout << "============Kokkos number of threads: " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;
    
    std::vector<std::vector<int>> csr_row_ptrs(length);
    std::vector<std::vector<int>> csr_col_idxs(length);
    std::vector<std::vector<double>> csr_values(length);
    std::vector<crs_matrix_type> sparse_csr_matrices(length);
    
    // 数据转换和矩阵创建
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        const auto& row_idx = std::get<0>(coo_list[i]);
        const auto& col_idx = std::get<1>(coo_list[i]);
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = row_idx.size();
        // Cast vals to double
        std::vector<double> vals(row_idx.size());
        const auto& vals_src = std::get<2>(coo_list[i]);
        for (size_t j = 0; j < vals_src.size(); ++j) {
            vals[j] = static_cast<double>(vals_src[j]);
        }

        coo2csr_new<int, double>(row_idx, col_idx, vals, rows, cols, csr_row_ptrs[i], csr_col_idxs[i], csr_values[i]);

        View<int*, DeviceType> row_ptr("row_ptr", csr_row_ptrs[i].size());
        View<int*, DeviceType> col_indices("col_indices", csr_col_idxs[i].size());
        View<double*, DeviceType> values("values", csr_values[i].size());

        Kokkos::deep_copy(row_ptr, Kokkos::View<int*, Kokkos::HostSpace>(csr_row_ptrs[i].data(), csr_row_ptrs[i].size()));
        Kokkos::deep_copy(col_indices, Kokkos::View<int*, Kokkos::HostSpace>(csr_col_idxs[i].data(), csr_col_idxs[i].size()));
        Kokkos::deep_copy(values, Kokkos::View<double*, Kokkos::HostSpace>(csr_values[i].data(), csr_values[i].size()));

        sparse_csr_matrices[i] = crs_matrix_type("A" + std::to_string(i), rows, cols, nnz, values, row_ptr, col_indices);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    
    if (order == "l2r") {
        // 简单的左到右乘法
        crs_matrix_type result = sparse_csr_matrices[0];
        
        for (int i = 1; i < length; ++i) {
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                int, int, double, device, device, device>;
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_SERIAL);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            // printf("Multiplying matrix %d\n", i);
            crs_matrix_type temp_result;
            
            try {
                KokkosSparse::spgemm_symbolic(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                KokkosSparse::spgemm_numeric(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                
                result = temp_result;
                
            } catch (const std::exception& e) {
                std::cerr << "Matrix multiplication failed at step " << i << ": " << e.what() << std::endl;
                failed_cases.insert("step_" + std::to_string(i));
                kh.destroy_spgemm_handle();
                return -1;
            }
            
            kh.destroy_spgemm_handle();
        }
        
        auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.values);
        result_nnz = result_values.extent(0);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        
        return result_nnz;
        
    } else {
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);
        
        result_nnz = multiply_by_order_kokkos_seq(order_vec, sparse_csr_matrices, failed_cases);
        
        return result_nnz;
    }
}
std::pair<long, long> multiply_by_order_kokkos_mul(std::vector<std::string>& order, 
                             std::vector<KokkosSparse::CrsMatrix<double, int, Kokkos::DefaultExecutionSpace, void, int>>& sparse_csr_matrices,
                             std::set<std::string>& failed_cases) {
    
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, device, device, device>;
    
    // std::cout << "\n=== Starting multiply_by_order_kokkos ===" << std::endl;
    
    std::map<std::string, crs_matrix_type> intermediate_results;
    std::set<std::string> initialized_keys;
    std::vector<long> num_mul;
    std::vector<long> inter_nnz;
    // 初始化原始矩阵
    for (int i = 0; i < sparse_csr_matrices.size(); ++i) {
        std::string matrix_name = "A" + std::to_string(i + 1);
        intermediate_results[matrix_name] = sparse_csr_matrices[i];
        initialized_keys.insert(matrix_name);
        // std::cout << "Initialized matrix: " << matrix_name << std::endl;
    }
    
    auto trim = [](const std::string& s) {
        auto start = s.find_first_not_of(" \t");
        auto end = s.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    };
    
    // 预分析引用计数
    std::map<std::string, int> ref_count;
    std::map<std::string, std::vector<int>> usage_steps;
    
    // std::cout << "\n=== Analyzing reference counts ===" << std::endl;
    for (int i = 0; i < order.size(); ++i) {
        const auto& expr = order[i];
        auto pos = expr.find('*');
        if (pos != std::string::npos) {
            std::string left = trim(expr.substr(0, pos));
            std::string right = trim(expr.substr(pos + 1));
            
            if (left.front() != 'A' || left.find('(') != std::string::npos) {
                usage_steps[left].push_back(i);
                ref_count[left]++;
            }
            if (right.front() != 'A' || right.find('(') != std::string::npos) {
                usage_steps[right].push_back(i);
                ref_count[right]++;
            }
        }
    }
    
    // 打印引用分析
    // for (const auto& pair : ref_count) {
    //     std::cout << "  \"" << pair.first << "\": used " << pair.second << " times" << std::endl;
    // }
    
    // std::cout << "\n=== Starting matrix multiplication sequence ===" << std::endl;
    
    for (int i = 0; i < order.size(); ++i) {
        std::string o = order[i];
        std::cout << "\n--- Processing order[" << i << "]: " << o << " ---" << std::endl;
        
        auto pos = o.find('*');
        if (pos == std::string::npos) {
            std::cout << "Skipping invalid expression: " << o << std::endl;
            continue;
        }
        
        std::string left = trim(o.substr(0, pos));
        std::string right = trim(o.substr(pos + 1));
        std::string result_key = "(" + left + " x " + right + ")";
        
        std::cout << "Left operand: \"" << left << "\"" << std::endl;
        std::cout << "Right operand: \"" << right << "\"" << std::endl;
        std::cout << "Result key: \"" << result_key << "\"" << std::endl;
        
        // 检查操作数是否存在
        if (initialized_keys.find(left) == initialized_keys.end()) {
            std::cerr << "ERROR: Left operand not found: " << left << std::endl;
            failed_cases.insert(o);
            return std::make_pair(-1, -1);
        }
        
        if (initialized_keys.find(right) == initialized_keys.end()) {
            std::cerr << "ERROR: Right operand not found: " << right << std::endl;
            failed_cases.insert(o);
            return std::make_pair(-1, -1);;
        }
        
        // 执行 Kokkos 矩阵乘法
        // std::cout << "Starting Kokkos matrix multiplication..." << std::endl;
        
        try {
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_KK);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            crs_matrix_type temp_result;
            
            // 符号阶段
            KokkosSparse::spgemm_symbolic(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // 数值阶段
            KokkosSparse::spgemm_numeric(
                kh, intermediate_results[left], false, 
                intermediate_results[right], false, temp_result
            );
            
            // std::cout << "Kokkos matrix multiplication completed successfully" << std::endl;
            // 计算temp_result矩阵所有元素的和
            auto temp_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), temp_result.values);
            double matrix_sum = 0.0;
            for (size_t j = 0; j < temp_values.extent(0); ++j) {
                matrix_sum += temp_values[j];
            }
            num_mul.push_back(static_cast<long>(matrix_sum));
            inter_nnz.push_back(temp_values.extent(0));
            // 将temp_result中的非零元素的值全部改为1
            Kokkos::parallel_for("set_values_to_one", temp_result.nnz(), KOKKOS_LAMBDA(const int i) {
                temp_result.values(i) = 1.0;
            });
            Kokkos::fence();
            // 存储结果
            intermediate_results[result_key] = temp_result;
            initialized_keys.insert(result_key);
            
            // 清理 kernel handle
            kh.destroy_spgemm_handle();
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Kokkos matrix multiplication failed: " << e.what() << std::endl;
            failed_cases.insert(o);
            return std::make_pair(-1, -1);;
        }
        
        // 基于引用计数的清理
        auto safe_cleanup = [&](const std::string& key) {
            if (key.front() != 'A' || key.find('(') != std::string::npos) {
                if (ref_count.count(key) && ref_count[key] > 0) {
                    ref_count[key]--;
                    // std::cout << "Decremented ref count for \"" << key 
                            //  << "\" to " << ref_count[key] << std::endl;
                    
                    if (ref_count[key] == 0) {
                        // std::cout << "Cleaning up: \"" << key << "\" (ref count reached 0)" << std::endl;
                        if (intermediate_results.count(key)) {
                            intermediate_results.erase(key);
                            initialized_keys.erase(key);
                        }
                    }
                }
            }
        };
        
        safe_cleanup(left);
        safe_cleanup(right);
        
        // std::cout << "Successfully computed: \"" << result_key << "\"" << std::endl;
        // std::cout << "Active matrices: " << initialized_keys.size() << std::endl;
        
        // 周期性清理检查
        if (i % 3 == 2) {
            std::cout << "Performing periodic cleanup check..." << std::endl;
            
            for (auto it = intermediate_results.begin(); it != intermediate_results.end();) {
                const std::string& key = it->first;
                bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
                
                if (!is_original && key != result_key) {
                    bool needed_soon = false;
                    for (int j = i + 1; j < std::min(i + 3, (int)order.size()); ++j) {
                        if (order[j].find(key) != std::string::npos) {
                            needed_soon = true;
                            break;
                        }
                    }
                    
                    if (!needed_soon) {
                        std::cout << "Force cleaning: \"" << key << "\"" << std::endl;
                        initialized_keys.erase(key);
                        if (ref_count.count(key)) {
                            ref_count[key] = 0;
                        }
                        it = intermediate_results.erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
    }
    
    // std::cout << "\n=== Finalizing results ===" << std::endl;
    
    // 获取最终结果
    if (order.empty()) {
        std::cerr << "ERROR: Empty order vector" << std::endl;
        return std::make_pair(-1, -1);;
    }
    
    std::string final_expr = order.back();
    auto final_pos = final_expr.find('*');
    if (final_pos == std::string::npos) {
        std::cerr << "ERROR: Invalid final expression: " << final_expr << std::endl;
        return std::make_pair(-1, -1);;
    }
    
    std::string final_left = trim(final_expr.substr(0, final_pos));
    std::string final_right = trim(final_expr.substr(final_pos + 1));
    std::string final_key = "(" + final_left + " x " + final_right + ")";
    
    std::cout << "Final result key: \"" << final_key << "\"" << std::endl;
    
    if (initialized_keys.find(final_key) == initialized_keys.end()) {
        std::cerr << "ERROR: Final result not found: " << final_key << std::endl;
        return std::make_pair(-1, -1);;
    }
    
    // 获取最终结果的非零元数量
    auto final_matrix = intermediate_results[final_key];
    auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_matrix.values);
    long result_nnz = result_values.extent(0);
    
    std::cout << "Final result nnz: " << result_nnz << std::endl;
    
    // 最终清理所有中间结果
    // std::cout << "\n=== Final cleanup ===" << std::endl;
    for (const auto& key : initialized_keys) {
        bool is_original = (key.front() == 'A' && key.find('(') == std::string::npos);
        if (!is_original) {
            std::cout << "Final cleanup: \"" << key << "\"" << std::endl;
        }
    }
    
    std::cout << "=== multiply_by_order_kokkos completed successfully ===" << std::endl;
    // 计算num_mul的和
    long total_mul = 0;
    for (const auto& val : num_mul) {
        total_mul += val;
    }

    // 获取inter_nnz的首个元素
    long first_inter_nnz = inter_nnz.empty() ? 0 : inter_nnz[0];

    std::cout << "Total multiplication sum: " << total_mul << std::endl;
    std::cout << "First intermediate nnz: " << first_inter_nnz << std::endl;

    // 返回值可以是total_mul + first_inter_nnz，或者根据需要修改
    // 这里假设返回total_mul
    return std::make_pair(total_mul, first_inter_nnz);
}

std::pair<long, long> smcm_coo_kokkos_mul(std::vector<std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>> coo_list,
                    std::vector<std::pair<int, int>> shape_list,
                    std::set<std::string>& failed_cases,
                    string order = "l2r",
                    int verbose = 0) {
    if (!Kokkos::is_initialized()) {
        std::cerr << "ERROR: Kokkos is not initialized. Please call Kokkos::initialize() in main()." << std::endl;
        return std::make_pair(-1, -1);;
    }
    using device = Kokkos::DefaultExecutionSpace;
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, device, void, int>;
    
    int length = coo_list.size();
    auto t0 = std::chrono::high_resolution_clock::now();
    long result_nnz = 0;
    std::cout << "Kokkos 实际使用线程数: " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;
    
    std::vector<std::vector<int>> csr_row_ptrs(length);
    std::vector<std::vector<int>> csr_col_idxs(length);
    std::vector<std::vector<double>> csr_values(length);
    std::vector<crs_matrix_type> sparse_csr_matrices(length);
    
    // 数据转换和矩阵创建
#pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        const auto& row_idx = std::get<0>(coo_list[i]);
        const auto& col_idx = std::get<1>(coo_list[i]);
        int rows = shape_list[i].first;
        int cols = shape_list[i].second;
        int nnz = row_idx.size();
        // Cast vals to double
        std::vector<double> vals(row_idx.size());
        const auto& vals_src = std::get<2>(coo_list[i]);
        for (size_t j = 0; j < vals_src.size(); ++j) {
            vals[j] = static_cast<double>(vals_src[j]);
        }

        coo2csr_new<int, double>(row_idx, col_idx, vals, rows, cols, csr_row_ptrs[i], csr_col_idxs[i], csr_values[i]);

        View<int*, DeviceType> row_ptr("row_ptr", csr_row_ptrs[i].size());
        View<int*, DeviceType> col_indices("col_indices", csr_col_idxs[i].size());
        View<double*, DeviceType> values("values", csr_values[i].size());

        Kokkos::deep_copy(row_ptr, Kokkos::View<int*, Kokkos::HostSpace>(csr_row_ptrs[i].data(), csr_row_ptrs[i].size()));
        Kokkos::deep_copy(col_indices, Kokkos::View<int*, Kokkos::HostSpace>(csr_col_idxs[i].data(), csr_col_idxs[i].size()));
        Kokkos::deep_copy(values, Kokkos::View<double*, Kokkos::HostSpace>(csr_values[i].data(), csr_values[i].size()));

        sparse_csr_matrices[i] = crs_matrix_type("A" + std::to_string(i), rows, cols, nnz, values, row_ptr, col_indices);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if (verbose > 0) {
        std::cout << "Time taken to create sparse matrices: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0
                  << " seconds" << std::endl;
    }
    
    if (order == "l2r") {
        // 简单的左到右乘法
        crs_matrix_type result = sparse_csr_matrices[0];
        
        for (int i = 1; i < length; ++i) {
            using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
                int, int, double, device, device, device>;
            KernelHandle kh;
            kh.create_spgemm_handle(KokkosSparse::SPGEMM_KK);
            kh.set_team_work_size(512);
            kh.set_dynamic_scheduling(true);
            
            // printf("Multiplying matrix %d\n", i);
            crs_matrix_type temp_result;
            
            try {
                KokkosSparse::spgemm_symbolic(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                KokkosSparse::spgemm_numeric(
                    kh, result, false, sparse_csr_matrices[i], false, temp_result
                );
                
                result = temp_result;
                
            } catch (const std::exception& e) {
                std::cerr << "Matrix multiplication failed at step " << i << ": " << e.what() << std::endl;
                failed_cases.insert("step_" + std::to_string(i));
                kh.destroy_spgemm_handle();
                return std::make_pair(-1, -1);;
            }
            
            kh.destroy_spgemm_handle();
        }
        
        auto result_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result.values);
        result_nnz = result_values.extent(0);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if (verbose > 0) {
            std::cout << "Final multiplied matrix non-zeros: " << result_nnz << std::endl;
            std::cout << "Time taken for final multiplication: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000000.0
                    << " seconds" << std::endl;
        }
        
        return std::make_pair(-1, -1);
        
    } else {
        std::cerr << "Computing order: " << order << std::endl;
        std::vector<std::string> order_vec;
        smcm_get_order(order, order_vec);

        auto [total_mul, first_inter_nnz] = multiply_by_order_kokkos_mul(order_vec, sparse_csr_matrices, failed_cases);

        return std::make_pair(total_mul, first_inter_nnz);
    }
}