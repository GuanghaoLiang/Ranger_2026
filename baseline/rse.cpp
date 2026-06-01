#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

namespace py = pybind11;

namespace {

struct LayerData {
	std::vector<int> row_ptr;
	std::vector<int> col_idx;
};

inline int compute_chunk_size(std::size_t work_items) {
	if (work_items == 0) {
		return 1;
	}
	int tnum = std::max(1, omp_get_max_threads());
	int denom = std::max(1, tnum * 8);
	int chunk = static_cast<int>(std::ceil(static_cast<double>(work_items) / denom));
	return std::max(1, chunk);
}

template <typename T>
std::vector<T> numpy_to_vector(py::array_t<T, py::array::c_style | py::array::forcecast> array) {
	py::buffer_info info = array.request();
	const T* ptr = static_cast<const T*>(info.ptr);
	return std::vector<T>(ptr, ptr + info.size);
}

LayerData coo_to_csr(const std::vector<int>& row_idx,
					 const std::vector<int>& col_idx,
					 int N) {
	if (row_idx.size() != col_idx.size()) {
		throw std::invalid_argument("Row and column arrays must have the same length");
	}
	LayerData layer;
	layer.row_ptr.assign(N + 1, 0);
	layer.col_idx.resize(col_idx.size());

	for (int row : row_idx) {
		if (row < 0 || row >= N) {
			throw std::out_of_range("Row index out of bounds");
		}
		layer.row_ptr[row + 1]++;
	}
	std::partial_sum(layer.row_ptr.begin(), layer.row_ptr.end(), layer.row_ptr.begin());

	std::vector<int> cursor = layer.row_ptr;
	for (std::size_t e = 0; e < row_idx.size(); ++e) {
		const int row = row_idx[e];
		const int insert_pos = cursor[row]++;
		layer.col_idx[insert_pos] = col_idx[e];
	}

	return layer;
}

std::vector<double> compute_row_sparsity(int N, const std::vector<int>& row_idx) {
	std::vector<double> sparsity(N, 0.0);
	for (int row : row_idx) {
		if (row >= 0 && row < N) {
			sparsity[row] += 1.0;
		}
	}
	if (N > 0) {
		const double inv = 1.0 / static_cast<double>(N);
		for (double& value : sparsity) {
			value *= inv;
		}
	}
	return sparsity;
}

std::vector<double> apply_chain_layer(const LayerData& layer,
									  const std::vector<double>& next_sparsity) {
	const int N = static_cast<int>(layer.row_ptr.size()) - 1;
	if (next_sparsity.size() != static_cast<std::size_t>(N)) {
		throw std::invalid_argument("Sparsity vector length mismatch");
	}
	std::vector<double> result(N, 0.0);
	const int chunk = compute_chunk_size(N);
	#pragma omp parallel for schedule(dynamic, chunk)
	for (int row = 0; row < N; ++row) {
		double rho = 1.0;
		for (int idx = layer.row_ptr[row]; idx < layer.row_ptr[row + 1]; ++idx) {
			const int col = layer.col_idx[idx];
			if (col < 0 || col >= N) {
				throw std::out_of_range("Column index out of bounds");
			}
			rho *= (1.0 - next_sparsity[col]);
		}
		result[row] = 1.0 - rho;
	}
	return result;
}

double mean_value(const std::vector<double>& values) {
	if (values.empty()) {
		return 0.0;
	}
	double sum = std::accumulate(values.begin(), values.end(), 0.0);
	return sum / static_cast<double>(values.size());
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> convert_index_list(const py::list& input_list) {
	std::vector<std::pair<std::vector<int>, std::vector<int>>> layers;
	layers.reserve(static_cast<std::size_t>(py::len(input_list)));
	for (auto item : input_list) {
		py::tuple tuple_item = py::cast<py::tuple>(item);
		auto first_array = tuple_item[0].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
		auto second_array = tuple_item[1].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
		layers.emplace_back(numpy_to_vector(first_array), numpy_to_vector(second_array));
	}
	return layers;
}

py::object rose_chain_cpp(int N, py::list index_list, bool full = false) {
	if (N <= 0) {
		throw std::invalid_argument("N must be positive");
	}
	auto layers_input = convert_index_list(index_list);
	const int p = static_cast<int>(layers_input.size());
	if (p == 0) {
		return py::float_(0.0);
	}

	std::vector<LayerData> csr_layers;
	csr_layers.reserve(p);
	for (const auto& layer : layers_input) {
		csr_layers.emplace_back(coo_to_csr(layer.first, layer.second, N));
	}

	if (!full) {
		std::vector<double> current = compute_row_sparsity(N, layers_input[p - 1].first);
		for (int layer = p - 2; layer >= 0; --layer) {
			current = apply_chain_layer(csr_layers[layer], current);
		}
		return py::float_(mean_value(current));
	}

	std::vector<std::vector<std::vector<double>>> R(p, std::vector<std::vector<double>>(p));
	std::vector<std::vector<double>> S(p, std::vector<double>(p, 0.0));

	for (int i = p - 1; i >= 0; --i) {
		R[i][i] = compute_row_sparsity(N, layers_input[i].first);
		S[i][i] = mean_value(R[i][i]);
		std::vector<double> current = R[i][i];
		for (int j = i - 1; j >= 0; --j) {
			current = apply_chain_layer(csr_layers[j], current);
			R[i][j] = current;
			S[i][j] = mean_value(current);
		}
	}

	py::array_t<double> R_out({p, p, N});
	py::array_t<double> S_out({p, p});

	auto R_buf = R_out.mutable_unchecked<3>();
	auto S_buf = S_out.mutable_unchecked<2>();
	for (int i = 0; i < p; ++i) {
		for (int j = 0; j < p; ++j) {
			S_buf(i, j) = S[i][j];
			if (j > i || R[i][j].empty()) {
				for (int n = 0; n < N; ++n) {
					R_buf(i, j, n) = 0.0;
				}
				continue;
			}
			for (int n = 0; n < N; ++n) {
				R_buf(i, j, n) = R[i][j][n];
			}
		}
	}

	return py::make_tuple(R_out, S_out);
}

}  // namespace

PYBIND11_MODULE(rse, m) {
	m.doc() = "Reduced sparsity estimator chain routines";
	m.def("rose_chain", &rose_chain_cpp,
		  "Compute Rose sparsity chain (returns mean when full=False)",
		  py::arg("N"), py::arg("index_list"), py::arg("full") = false);
}

