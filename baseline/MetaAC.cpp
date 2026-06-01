#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace std;

// AC-style sparse chain density proxy (matches sm25/baseline/Meta.py::MetaAC).
vector<double> meta_ac_chain(int N, const vector<pair<vector<int>, vector<int>>>& index_list) {
    const int k = static_cast<int>(index_list.size());
    if (k == 0) {
        return {};
    }
    const double nn = static_cast<double>(N) * static_cast<double>(N);
    vector<double> sp(k, 0.0);
    sp[0] = static_cast<double>(index_list[0].first.size()) / nn;
    for (int i = 1; i < k; ++i) {
        const double lens_i = static_cast<double>(index_list[i].first.size());
        sp[i] = static_cast<double>(N) * (lens_i / nn) * sp[i - 1];
    }
    return sp;
}

py::array_t<double> meta_ac_chain_wrapper(int N, py::list index_list) {
    vector<pair<vector<int>, vector<int>>> vec_index_list;
    vec_index_list.reserve(index_list.size());
    for (auto item : index_list) {
        py::tuple tuple_item = py::cast<py::tuple>(item);
        py::array_t<int> row = py::cast<py::array_t<int>>(tuple_item[0]);
        py::array_t<int> col = py::cast<py::array_t<int>>(tuple_item[1]);
        vector<int> row_vec(row.size());
        vector<int> col_vec(col.size());
        memcpy(row_vec.data(), row.data(), row.size() * sizeof(int));
        memcpy(col_vec.data(), col.data(), col.size() * sizeof(int));
        vec_index_list.emplace_back(std::move(row_vec), std::move(col_vec));
    }

    vector<double> sp = meta_ac_chain(N, vec_index_list);
    py::array_t<double> out(sp.size());
    if (!sp.empty()) {
        memcpy(out.mutable_data(), sp.data(), sp.size() * sizeof(double));
    }
    return out;
}

PYBIND11_MODULE(MetaAC, m) {
    m.doc() = "MetaAC sparse chain density proxy (AC-style chaining)";
    m.def("MetaAC", &meta_ac_chain_wrapper,
          "Layer sparsities from consecutive AC-style chaining",
          py::arg("N"), py::arg("index_list"));
}
