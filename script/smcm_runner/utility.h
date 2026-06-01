#ifndef _UTILITY_H
#define _UTILITY_H

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <map>
#include <mkl.h>
#include <iomanip>
//#include <tbb/scalable_allocator.h>

using namespace std;
#define EPSILON 0.001

template <class T>
struct ErrorTolerantEqual {
  ErrorTolerantEqual(const T &myepsilon) : epsilon(myepsilon) {}
  inline bool operator()(const T &a, const T &b) const {
    if (a == b)
      return true;

    return (std::abs(a - b) < epsilon ||
            (std::abs(a - b) / std::max(std::abs(a), std::abs(b))) < epsilon);
  }
  T epsilon;
};

// Because identify reports ambiguity in PGI compilers
template <typename T>
struct myidentity {
  const T operator()(const T &x) const { return x; }
};

template <typename _ForwardIterator, typename _StrictWeakOrdering>
bool my_is_sorted(_ForwardIterator __first, _ForwardIterator __last,
                  _StrictWeakOrdering __comp) {
  if (__first == __last)
    return true;

  _ForwardIterator __next = __first;
  for (++__next; __next != __last; __first = __next, ++__next)
    if (__comp(*__next, *__first))
      return false;
  return true;
};

template <typename ITYPE> ITYPE CumulativeSum(ITYPE *arr, ITYPE size) {
  ITYPE prev;
  ITYPE tempnz = 0;
  for (ITYPE i = 0; i < size; ++i) {
    prev = arr[i];
    arr[i] = tempnz;
    tempnz += prev;
  }
  return (tempnz); // return sum
}

template <typename _ForwardIter, typename T>
void iota(_ForwardIter __first, _ForwardIter __last, T __value) {
  while (__first != __last)
    *__first++ = __value++;
}

template <typename T, typename I> T **allocate2D(I m, I n) {
  T **array = new T *[m];
  for (I i = 0; i < m; ++i)
    array[i] = new T[n];
  return array;
}

template <typename T, typename I> void deallocate2D(T **array, I m) {
  for (I i = 0; i < m; ++i)
    delete[] array[i];
  delete[] array;
}

template <typename T> struct absdiff {
  T operator()(T const &arg1, T const &arg2) const {
    using std::abs;
    return abs(arg1 - arg2);
  }
};

/* This function will return n % d.
   d must be one of: 1, 2, 4, 8, 16, 32, … */
inline unsigned int getModulo(unsigned int n, unsigned int d) {
  return (n & (d - 1));
}

// Same requirement (d=2^k) here as well
inline unsigned int getDivident(unsigned int n, unsigned int d) {
  while ((d = d >> 1))
    n = n >> 1;
  return n;
}

// Memory allocation by C++-new / Aligned malloc / scalable malloc
template <typename T> inline T *my_malloc(int array_size) {
#ifdef CPP
   T * a = new T[array_size];
  #pragma omp parallel for
  for(int i=0; i<array_size; i++)
  {
	a[i] = T();
  } 
  return a;
#elif defined IMM
  return (T *)_mm_malloc(sizeof(T) * array_size, 64);
#endif
}

// Memory deallocation
template <typename T> inline void my_free(T *a) {
#ifdef CPP
  delete[] a;
#elif defined IMM
  _mm_free(a);
#endif
}

// Prefix sum (Sequential)
template <typename T> void seq_scan(T *in, T *out, T N) {
  out[0] = 0;
  for (T i = 0; i < N - 1; ++i) {
    out[i + 1] = out[i] + in[i];
  }
}

// Prefix sum (Thread parallel)
template <typename T> void scan(T *in, T *out, T N) {
  // if the array is comparatively small, use sequential scan instead
  if (N < (1 << 17)) {
    seq_scan(in, out, N);
  } else {
    int tnum = 1;
#pragma omp parallel
    { tnum = omp_get_num_threads(); }
    T each_n = N / tnum;
    T *partial_sum = my_malloc<T>(tnum);
#pragma omp parallel
    {
      // thead level prefix summing
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1) ? start + each_n : N;
      out[start] = 0;
      for (T i = start; i < end - 1; ++i) {
        out[i + 1] = out[i] + in[i];
      }
      // calculate offset in every thread
      partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier

      T offset = 0;
      for (int i = 0; i < tid; ++i) {
        offset += partial_sum[i];
      }
      for (T i = start; i < end; ++i) {
        out[i] += offset;
      }
    }
    my_free<T>(partial_sum);
  }
}

// Sort by key
template <typename IT, typename NT>
inline void mergesort(IT *nnz_num, NT *nnz_sorting, IT *temp_num,
                      NT *temp_sorting, IT left, IT right) {
  int mid, i, j, k;

  if (left >= right) {
    return;
  }

  mid = (left + right) / 2;

  mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, left, mid);
  mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, mid + 1, right);

  for (i = left; i <= mid; ++i) {
    temp_num[i] = nnz_num[i];
    temp_sorting[i] = nnz_sorting[i];
  }

  for (i = mid + 1, j = right; i <= right; ++i, --j) {
    temp_sorting[i] = nnz_sorting[j];
    temp_num[i] = nnz_num[j];
  }

  i = left;
  j = right;

  for (k = left; k <= right; ++k) {
    if (temp_num[i] <= temp_num[j] && i <= mid) {
      nnz_num[k] = temp_num[i];
      nnz_sorting[k] = temp_sorting[i++];
    } else {
      nnz_num[k] = temp_num[j];
      nnz_sorting[k] = temp_sorting[j--];
    }
  }
}

// Sorting key-value
template <typename IT, typename NT>
inline void cpu_sorting_key_value(IT *key, NT *value, IT N) {
  IT *temp_key;
  NT *temp_value;

  temp_key = my_malloc<IT>(N);
  temp_value = my_malloc<NT>(N);

  mergesort(key, value, temp_key, temp_value, 0, N - 1);

  my_free<IT>(temp_key);
  my_free<NT>(temp_value);
}


long get_nnz_from_csr(sparse_matrix_t A) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols, *rows_start, *rows_end, *col_indx, info;
    double* values;

    // 导出 CSR 格式数据
    mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
    // 非零元个数 = 最后一行结束位置 - 第一行开始位置
    long nnz = rows_end[rows - 1] - rows_start[0];

    return nnz;
}

int get_nnz_from_csc(sparse_matrix_t A) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols, *cols_start, *cols_end, *row_indx, info;
    double* values;

    // 导出 CSC 格式数据
    mkl_sparse_d_export_csc(A, &indexing, &rows, &cols, &cols_start, &cols_end, &row_indx, &values);
    // 非零元个数 = 最后一列结束位置 - 第一列开始位置
    printf("===get_nnz_from_csc: Exported CSC matrix with %d rows and %d cols\n", rows, cols);
    int nnz = cols_end[cols - 1] - cols_start[0];
    printf("bang");
    return nnz;
} 

int catalan_number(int n) {
    int _n = n-1;
    if (_n <= 1) return 1;
    std::vector<long long> catalan(_n + 1, 0);
    catalan[0] = catalan[1] = 1;
    for (int i = 2; i <= _n; ++i) {
        for (int j = 0; j < i; ++j) {
            catalan[i] += catalan[j] * catalan[i - 1 - j];
        }
    }
    return static_cast<int>(catalan[_n]);
}

void generate_parentheses(int start, int end, std::vector<std::string>& result) {
    if (start == end) {
        result.push_back("A" + std::to_string(start));
        return;
    }
    for (int i = start; i < end; ++i) {
        std::vector<std::string> left, right;
        generate_parentheses(start, i, left);
        generate_parentheses(i + 1, end, right);
        for (const auto& l : left) {
            for (const auto& r : right) {
                result.push_back("(" + l + " x " + r + ")");
            }
        }
    }
}
std::pair<std::string, std::string> split_expression(const std::string& expr) {
    int stack = 0;
    for (size_t i = 0; i < expr.size(); ++i) {
        if (expr[i] == '(') stack++;
        else if (expr[i] == ')') stack--;
        else if (expr[i] == 'x' && stack == 1) {
            // 左右去掉外层括号
            return {expr.substr(1, i - 2), expr.substr(i + 2, expr.size() - i - 3)};
        }
    }
    return {"", ""};
}

void smcm_get_order(const std::string& expr, std::vector<std::string>& order, int depth = 0) {
    std::string indent(depth * 2, ' ');
    if (expr.find('x') == std::string::npos) {
        // std::cout << indent << "eval_parentheses(\"" << expr << "\") = " << expr << std::endl;
        return;
    }
    // std::cout << indent << "eval_parentheses(\"" << expr << "\")" << std::endl;
    auto [left, right] = split_expression(expr);
    smcm_get_order(left, order, depth + 1);
    smcm_get_order(right, order, depth + 1);
    // std::cout << indent << "-> " << left << " * " << right << std::endl;
    std::string tmp = left + " * " + right;
    order.push_back(tmp);
}
std::string formatDouble(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    return oss.str();
}

// 将 entry 格式化为指定字符串格式
std::string formatEntry(const std::string& id, double range, const std::map<std::string, double>& attrs) {
    std::ostringstream oss;

    // 输出 id 和 range
    oss << id << " " << formatDouble(range) << " [";

    // 输出 key:value 列表
    bool first = true;
    for (const auto& [key, value] : attrs) {
        if (!first) {
            oss << ", ";
        }
        oss << key << ":" << formatDouble(value);
        first = false;
    }

    oss << "]";
    return oss.str();
}


// query cpu cache
// size_t i386_cpuid_caches() {
//   int i;
//   size_t total_avail_cache = 0;

//   for (i = 0; i < 32; i++) {

//     // Variables to hold the contents of the 4 i386 legacy registers
//     uint32_t eax, ebx, ecx, edx;

//     eax = 4; // get cache info
//     ecx = i; // cache id

//     __asm__ __volatile__(
//         "cpuid"     // call i386 cpuid instruction
//         : "+a"(eax), "=b"(ebx), "+c"(ecx), "=d"(edx) // generates output in 4 registers eax, ebx, ecx and edx
//         : "a"(eax), "c"(ecx));

//     // taken from http://download.intel.com/products/processor/manual/325462.pdf
//     // Vol. 2A 3-149
//     int cache_type = eax & 0x1F;

//     if (cache_type == 0) // end of valid cache identifiers
//       break;

//     char const *cache_type_string;
//     switch (cache_type) {
//     case 1:
//       cache_type_string = "Data Cache";
//       break;
//     case 2:
//       cache_type_string = "Instruction Cache";
//       break;
//     case 3:
//       cache_type_string = "Unified Cache";
//       break;
//     default:
//       cache_type_string = "Unknown Type Cache";
//       break;
//     }

//     int cache_level = (eax >>= 5) & 0x7;

//     int cache_is_self_initializing =
//         (eax >>= 3) & 0x1; // does not need SW initialization
//     int cache_is_fully_associative = (eax >>= 1) & 0x1;

//     // taken from http://download.intel.com/products/processor/manual/325462.pdf
//     // 3-166 Vol. 2A ebx contains 3 integers of 10, 10 and 12 bits respectively
//     unsigned int cache_sets = ecx + 1;
//     unsigned int cache_coherency_line_size = (ebx & 0xFFF) + 1;
//     unsigned int cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;
//     unsigned int cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;

//     // Total cache size is the product
//     size_t cache_total_size = cache_ways_of_associativity *
//                               cache_physical_line_partitions *
//                               cache_coherency_line_size * cache_sets;

//     if (cache_type == 1 or cache_type == 3)
//       total_avail_cache = std::max(total_avail_cache, cache_total_size);
//   }

//   return total_avail_cache;
// }

#endif
