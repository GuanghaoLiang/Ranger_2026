// Ranger13 — Ranger12 with bitmap-native coolfull pruning (no std::set).
//
// Same coolfull semantics as Ranger12:
//   forward:  common = right_nrows[i+1] ∩ col_idx[i]
//   backward: common = row_idx[i] ∩ left_ncols[i-1]
//
// Optimizations vs Ranger12:
//   - right_nrows / col_idx / row_idx / left_ncols stored as BitSet
//   - intersection via padded word-wise AND (no set_intersection)
//   - survivor indices merged with bitset_or_into instead of set::insert
//
// remap_chain / compute_pipeline are unchanged from Ranger7.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <omp.h>

namespace py = pybind11;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr int kBitsPerWord = 64;

using BitWord = std::uint64_t;

struct FeatureRing {
    std::array<std::vector<double>, 2> buffers;
    int active = 0;

    std::vector<double>& current() { return buffers[active]; }
    std::vector<double>& next() { return buffers[active ^ 1]; }
    void rotate() { active ^= 1; }
};

struct EdgeWorkspace {
    std::vector<int> offsets;
    std::vector<int> adjacency;
    std::vector<int> scratch;
};

// ---------------------------------------------------------------------
// Packed bitset, sized in 64-bit words but probed via 32-bit gathers
// so the AVX-512 inner loop can feed `vpgatherdd` directly.  The
// underlying storage is over-allocated by one extra 64-bit word so a
// `vpgatherdd` indexed at `(M-1) >> 5` never reads past the end.
// ---------------------------------------------------------------------
class BitSet {
public:
    void reset(std::size_t n_bits) {
        size_bits_ = n_bits;
        n_words_ = (n_bits + kBitsPerWord - 1) / kBitsPerWord;
        // One extra word of headroom for safe 32-bit gather past the
        // logical end.
        const std::size_t storage = n_words_ + 1;
        if (words_.size() < storage) {
            words_.resize(storage);
        }
        std::fill(words_.begin(), words_.begin() + storage, 0ULL);
    }

    BitWord* data() { return words_.data(); }
    const BitWord* data() const { return words_.data(); }
    const std::uint32_t* data_u32() const {
        return reinterpret_cast<const std::uint32_t*>(words_.data());
    }
    std::size_t n_words() const { return n_words_; }
    std::size_t size_bits() const { return size_bits_; }

private:
    std::vector<BitWord> words_;
    std::size_t n_words_ = 0;
    std::size_t size_bits_ = 0;
};

inline int parallel_max(const std::vector<int>& v) {
    const std::size_t n = v.size();
    if (n == 0) {
        return -1;
    }
    int m = std::numeric_limits<int>::min();
    #pragma omp parallel for schedule(static) reduction(max: m)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        if (v[i] > m) {
            m = v[i];
        }
    }
    return m;
}

// Atomically OR each id's bit into `dst`.  On Ice Lake the locked OR
// throughput is ~5–10 cycles uncontended; the bit set is sized to fit
// in L2/L3 so contention stays at the cache-line level rather than at
// memory.
inline void bitset_or_into(BitSet& dst, const std::vector<int>& ids) {
    BitWord* bits = dst.data();
    const std::size_t n = ids.size();
    if (n == 0) {
        return;
    }
    #pragma omp parallel for schedule(static, 4096)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        const std::uint32_t id = static_cast<std::uint32_t>(ids[i]);
        const std::size_t w = id >> 6;
        const BitWord bit = 1ULL << (id & 63);
        __atomic_fetch_or(&bits[w], bit, __ATOMIC_RELAXED);
    }
}

// AVX-512 vectorised AND of two equal-sized bitsets.
inline void bitset_and(BitSet& out, const BitSet& a, const BitSet& b) {
    const std::size_t nw = a.n_words();
    BitWord* o = out.data();
    const BitWord* pa = a.data();
    const BitWord* pb = b.data();
    const std::size_t nv = nw / 8;
    #pragma omp parallel for schedule(static)
    for (long v = 0; v < static_cast<long>(nv); ++v) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pa + v * 8));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pb + v * 8));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(o + v * 8),
                            _mm512_and_si512(va, vb));
    }
    for (std::size_t w = nv * 8; w < nw; ++w) {
        o[w] = pa[w] & pb[w];
    }
}

// Parallel popcount over the active prefix of a bitset.  On Ice Lake
// `_mm512_popcnt_epi64` is one cycle per 8 uint64s.
inline std::size_t bitset_popcount(const BitSet& bs) {
    const BitWord* p = bs.data();
    const std::size_t nw = bs.n_words();
    long long total = 0;
    const std::size_t nv = nw / 8;
    #pragma omp parallel for schedule(static) reduction(+: total)
    for (long v = 0; v < static_cast<long>(nv); ++v) {
        __m512i x = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p + v * 8));
        __m512i c = _mm512_popcnt_epi64(x);
        total += _mm512_reduce_add_epi64(c);
    }
    long long tail = 0;
    for (std::size_t w = nv * 8; w < nw; ++w) {
        tail += __builtin_popcountll(p[w]);
    }
    return static_cast<std::size_t>(total + tail);
}

// 16-way bitmap test.  Each lane: bitmap[id >> 5] >> (id & 31) & 1.
static inline __mmask16 bitmap_test_16(const std::uint32_t* bitmap32, __m512i ids) {
    __m512i word_idx = _mm512_srli_epi32(ids, 5);
    __m512i words = _mm512_i32gather_epi32(word_idx,
                                           reinterpret_cast<const int*>(bitmap32),
                                           4);
    __m512i bit_pos = _mm512_and_si512(ids, _mm512_set1_epi32(31));
    __m512i bit_mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), bit_pos);
    return _mm512_test_epi32_mask(words, bit_mask);
}

// Two-pass parallel stream compaction with AVX-512 `vpcompressd`.
//
// `ids_idx` selects which of `a`, `b` is the membership-test source:
//   0 -> use a (compact based on a[i] in bitmap)
//   1 -> use b (compact based on b[i] in bitmap)
// Both `a` and `b` are compacted lockstep using the same per-element
// keep/discard decision.  Returns the survivor count.
//
// When the survivor count equals the input size, the function leaves
// `a`/`b` unchanged (no allocation, no second pass).
std::size_t compact_pair_with_bitmap(std::vector<int>& a, std::vector<int>& b,
                                     int ids_idx,
                                     const std::uint32_t* bitmap32) {
    const std::size_t n = a.size();
    if (n == 0) {
        return 0;
    }
    const int* ids = (ids_idx == 0 ? a.data() : b.data());

    int max_threads = std::max(1, omp_get_max_threads());
    int blocks = std::max(1, std::min<int>(max_threads,
        static_cast<int>((n + 16383) / 16384)));
    if (n < 16384) {
        blocks = 1;
    }

    std::vector<std::size_t> starts(blocks + 1);
    starts[0] = 0;
    for (int t = 1; t < blocks; ++t) {
        std::size_t s = (n * static_cast<std::size_t>(t)) / static_cast<std::size_t>(blocks);
        // Round inner block boundaries down to multiples of 16 so each
        // block can run a clean vectorised loop.
        s &= ~static_cast<std::size_t>(15);
        starts[t] = s;
    }
    starts[blocks] = n;

    std::vector<std::size_t> counts(blocks, 0);

    auto count_block = [&](std::size_t s, std::size_t e) -> std::size_t {
        std::size_t c = 0;
        std::size_t i = s;
        while (i + 16 <= e) {
            __m512i v_ids = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(ids + i));
            __mmask16 m = bitmap_test_16(bitmap32, v_ids);
            c += static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned>(m)));
            i += 16;
        }
        for (; i < e; ++i) {
            const std::uint32_t id = static_cast<std::uint32_t>(ids[i]);
            const std::uint32_t w = bitmap32[id >> 5];
            if ((w >> (id & 31)) & 1u) {
                ++c;
            }
        }
        return c;
    };

    if (blocks == 1) {
        counts[0] = count_block(0, n);
    } else {
        #pragma omp parallel num_threads(blocks)
        {
            const int tid = omp_get_thread_num();
            counts[tid] = count_block(starts[tid], starts[tid + 1]);
        }
    }

    std::vector<std::size_t> bases(blocks + 1, 0);
    for (int t = 0; t < blocks; ++t) {
        bases[t + 1] = bases[t] + counts[t];
    }
    const std::size_t total = bases[blocks];

    if (total == n) {
        return total;
    }

    std::vector<int> out_a(total);
    std::vector<int> out_b(total);

    auto compact_block = [&](std::size_t s, std::size_t e, std::size_t w_start) {
        std::size_t w = w_start;
        std::size_t i = s;
        while (i + 16 <= e) {
            __m512i v_ids = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(ids + i));
            __mmask16 m = bitmap_test_16(bitmap32, v_ids);
            __m512i v_a = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(a.data() + i));
            __m512i v_b = _mm512_loadu_si512(
                reinterpret_cast<const __m512i*>(b.data() + i));
            _mm512_mask_compressstoreu_epi32(
                reinterpret_cast<void*>(out_a.data() + w), m, v_a);
            _mm512_mask_compressstoreu_epi32(
                reinterpret_cast<void*>(out_b.data() + w), m, v_b);
            w += static_cast<std::size_t>(__builtin_popcount(static_cast<unsigned>(m)));
            i += 16;
        }
        for (; i < e; ++i) {
            const std::uint32_t id = static_cast<std::uint32_t>(ids[i]);
            const std::uint32_t word = bitmap32[id >> 5];
            if ((word >> (id & 31)) & 1u) {
                out_a[w] = a[i];
                out_b[w] = b[i];
                ++w;
            }
        }
    };

    if (blocks == 1) {
        compact_block(0, n, 0);
    } else {
        #pragma omp parallel num_threads(blocks)
        {
            const int tid = omp_get_thread_num();
            compact_block(starts[tid], starts[tid + 1], bases[tid]);
        }
    }

    a.swap(out_a);
    b.swap(out_b);
    return total;
}

// Sentinel: refuse the bitmap path when the universe would be huge.
// 1 << 31 bits = 256 MiB per bitmap × 3 bitmaps = 768 MiB, which is
// already more than we'd want.  Practical HIN node-id ranges sit
// orders of magnitude below this.
constexpr std::size_t kMaxBitmapBits = (std::size_t{1} << 31);

inline void bitset_and_resized(BitSet& out, const BitSet& a, const BitSet& b) {
    const std::size_t M = std::max(a.size_bits(), b.size_bits());
    if (M == 0) {
        throw std::runtime_error("No common element between consecutive layers.");
    }
    if (M > kMaxBitmapBits) {
        throw std::runtime_error(
            "Ranger13: bitmap universe exceeds 2^31 bits; rebuild with a"
            " hash-based fallback if needed.");
    }
    out.reset(M);
    const BitWord* pa = a.data();
    const BitWord* pb = b.data();
    BitWord* o = out.data();
    const std::size_t nwa = a.n_words();
    const std::size_t nwb = b.n_words();
    const std::size_t nw = out.n_words();
    for (std::size_t w = 0; w < nw; ++w) {
        const BitWord wa = (w < nwa) ? pa[w] : 0ULL;
        const BitWord wb = (w < nwb) ? pb[w] : 0ULL;
        o[w] = wa & wb;
    }
    if (bitset_popcount(out) == 0) {
        throw std::runtime_error("No common element between consecutive layers.");
    }
}

inline void bitset_from_survivors(BitSet& dst,
                                  const std::vector<int>& probe,
                                  const std::vector<int>& values,
                                  const std::uint32_t* bitmap32) {
    const std::size_t n = probe.size();
    if (n == 0) {
        dst.reset(0);
        return;
    }
    std::vector<int> survivors;
    survivors.reserve(n / 4 + 1);
    int max_id = -1;
    for (std::size_t j = 0; j < n; ++j) {
        const std::uint32_t id = static_cast<std::uint32_t>(probe[j]);
        const std::uint32_t word = bitmap32[id >> 5];
        if ((word >> (id & 31)) & 1u) {
            const int val = values[j];
            survivors.push_back(val);
            if (val > max_id) {
                max_id = val;
            }
        }
    }
    if (max_id < 0) {
        dst.reset(0);
        return;
    }
    dst.reset(static_cast<std::size_t>(max_id) + 1);
    bitset_or_into(dst, survivors);
}

inline void prune_side_with_bitmap(std::pair<std::vector<int>, std::vector<int>>& side,
                                     int ids_idx,
                                     const std::uint32_t* bitmap32) {
    if (ids_idx == 0) {
        compact_pair_with_bitmap(side.first, side.second, /*ids_idx=*/0, bitmap32);
    } else {
        compact_pair_with_bitmap(side.first, side.second, /*ids_idx=*/1, bitmap32);
    }
}

std::vector<BitSet> build_right_nrows_bm(
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& index_list) {
    const int layers = static_cast<int>(index_list.size());
    std::vector<BitSet> right_nrows(layers);
    for (int i = 0; i < layers; ++i) {
        const std::vector<int>& ids =
            (i == 0) ? index_list[i].second : index_list[i].first;
        const int max_id = parallel_max(ids);
        if (max_id < 0) {
            right_nrows[i].reset(0);
            continue;
        }
        right_nrows[i].reset(static_cast<std::size_t>(max_id) + 1);
        bitset_or_into(right_nrows[i], ids);
    }
    return right_nrows;
}

void cool_filter_coolfull_bidirectional(
    std::vector<std::pair<std::vector<int>, std::vector<int>>>& index_list,
    const std::vector<BitSet>& right_nrows) {
    const int layers = static_cast<int>(index_list.size());
    if (layers <= 1) {
        return;
    }
    if (static_cast<int>(right_nrows.size()) != layers) {
        throw std::invalid_argument("right_nrows size must match index_list size.");
    }

    std::vector<BitSet> col_idx(layers);
    std::vector<BitSet> row_idx(layers);
    std::vector<BitSet> left_ncols(layers);
    col_idx[0] = right_nrows[0];

    BitSet common;

    for (int i = 0; i < layers - 1; ++i) {
        auto& left = index_list[i];
        auto& right = index_list[i + 1];
        if (left.second.empty() || right.first.empty()) {
            throw std::runtime_error("No common element between consecutive layers.");
        }

        bitset_and_resized(left_ncols[i], right_nrows[i + 1], col_idx[i]);
        const std::uint32_t* bm32 = left_ncols[i].data_u32();

        const std::vector<int> right_join_orig = right.first;
        const std::vector<int> left_join_orig = left.second;
        const std::vector<int> tmp = right.second;
        const std::vector<int> tmp2 = left.first;

        prune_side_with_bitmap(left, /*ids_idx=*/1, bm32);
        prune_side_with_bitmap(right, /*ids_idx=*/0, bm32);

        bitset_from_survivors(col_idx[i + 1], right_join_orig, tmp, bm32);
        bitset_from_survivors(row_idx[i], left_join_orig, tmp2, bm32);
    }

    for (int i = layers - 2; i >= 1; --i) {
        auto& left_layer = index_list[i - 1];
        auto& right_layer = index_list[i];
        if (left_layer.second.empty() || right_layer.first.empty()) {
            throw std::runtime_error("No common element between consecutive layers.");
        }

        bitset_and_resized(common, row_idx[i], left_ncols[i - 1]);
        const std::uint32_t* bm32 = common.data_u32();

        const std::vector<int> left_join_orig = left_layer.second;
        const std::vector<int> tmp = left_layer.first;

        prune_side_with_bitmap(right_layer, /*ids_idx=*/0, bm32);
        prune_side_with_bitmap(left_layer, /*ids_idx=*/1, bm32);

        bitset_from_survivors(row_idx[i - 1], left_join_orig, tmp, bm32);
    }
}

void cool_filter_bidirectional(std::vector<std::pair<std::vector<int>, std::vector<int>>>& index_list) {
    const auto right_nrows = build_right_nrows_bm(index_list);
    cool_filter_coolfull_bidirectional(index_list, right_nrows);
}

// ---------------------------------------------------------------------
// Below this point: copied verbatim from Ranger7 / Ranger6.
// ---------------------------------------------------------------------

std::pair<std::vector<int>, int> compress_indices(const std::vector<int>& data) {
    const std::size_t n = data.size();
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        pairs.emplace_back(data[i], static_cast<int>(i));
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    std::vector<int> compressed(n);
    int label = -1;
    int last_value = std::numeric_limits<int>::min();
    for (const auto& kv : pairs) {
        if (label == -1 || kv.first != last_value) {
            ++label;
            last_value = kv.first;
        }
        compressed[kv.second] = label;
    }
    return {std::move(compressed), label + 1};
}

std::tuple<std::vector<int>, std::vector<int>, int> dual_compress(const std::vector<int>& left,
                                                                  const std::vector<int>& right) {
    std::vector<int> all;
    all.reserve(left.size() + right.size());
    all.insert(all.end(), left.begin(), left.end());
    all.insert(all.end(), right.begin(), right.end());
    auto comp = compress_indices(all);
    const std::vector<int>& compressed = comp.first;
    std::vector<int> left_comp(left.size());
    std::vector<int> right_comp(right.size());
    std::copy(compressed.begin(), compressed.begin() + static_cast<long>(left.size()), left_comp.begin());
    std::copy(compressed.begin() + static_cast<long>(left.size()), compressed.end(), right_comp.begin());
    return {std::move(left_comp), std::move(right_comp), comp.second};
}

std::vector<int> remap_chain(std::vector<std::pair<std::vector<int>, std::vector<int>>>& index_list) {
    const int layers = static_cast<int>(index_list.size());
    std::vector<int> n_nodes;
    n_nodes.assign(layers + 1, 0);
    if (layers == 0) {
        return n_nodes;
    }

    auto head = compress_indices(index_list[0].first);
    index_list[0].first = std::move(head.first);
    n_nodes[0] = head.second;

    for (int layer = 0; layer < layers - 1; ++layer) {
        auto tmp = dual_compress(index_list[layer].second, index_list[layer + 1].first);
        index_list[layer].second = std::move(std::get<0>(tmp));
        index_list[layer + 1].first = std::move(std::get<1>(tmp));
        n_nodes[layer + 1] = std::get<2>(tmp);
    }

    auto tail = compress_indices(index_list[layers - 1].second);
    index_list[layers - 1].second = std::move(tail.first);
    n_nodes[layers] = tail.second;
    return n_nodes;
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> convert_to_vector_pair(const py::list& input_list) {
    std::vector<std::pair<std::vector<int>, std::vector<int>>> index_list;
    index_list.reserve(static_cast<std::size_t>(py::len(input_list)));
    for (auto item : input_list) {
        py::tuple tuple_item = py::cast<py::tuple>(item);
        py::array_t<int, py::array::c_style | py::array::forcecast> first_array =
            tuple_item[0].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        py::array_t<int, py::array::c_style | py::array::forcecast> second_array =
            tuple_item[1].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        std::vector<int> first_vector(first_array.size());
        std::vector<int> second_vector(second_array.size());
        if (first_array.size() > 0) {
            std::memcpy(first_vector.data(), first_array.data(), first_array.size() * sizeof(int));
        }
        if (second_array.size() > 0) {
            std::memcpy(second_vector.data(), second_array.data(), second_array.size() * sizeof(int));
        }
        index_list.emplace_back(std::move(first_vector), std::move(second_vector));
    }
    return index_list;
}

void build_adjacency(const std::vector<int>& src_idx,
                     const std::vector<int>& dst_idx,
                     int dst_nodes,
                     EdgeWorkspace& workspace) {
    workspace.offsets.assign(dst_nodes + 1, 0);
    for (int dst : dst_idx) {
        if (dst < 0 || dst >= dst_nodes) {
            throw std::runtime_error("Destination index out of bounds.");
        }
        workspace.offsets[dst + 1]++;
    }
    std::partial_sum(workspace.offsets.begin(), workspace.offsets.end(), workspace.offsets.begin());

    workspace.adjacency.resize(dst_idx.size());
    workspace.scratch = workspace.offsets;
    for (std::size_t e = 0; e < src_idx.size(); ++e) {
        const int dst = dst_idx[e];
        const int pos = workspace.scratch[dst]++;
        workspace.adjacency[pos] = src_idx[e];
    }
}

double compute_pipeline(std::vector<std::pair<std::vector<int>, std::vector<int>>>& index_list,
                        const std::vector<int>& n_nodes,
                        int r) {
    if (index_list.empty()) {
        return 0.0;
    }
    if (r <= 0) {
        throw std::invalid_argument("Parameter r must be positive.");
    }

    const int dims = r * 2;
    FeatureRing features;
    EdgeWorkspace workspace;
    const int layers = static_cast<int>(index_list.size());

    const std::size_t head_elems = static_cast<std::size_t>(n_nodes[0]) * dims;
    if (features.current().size() < head_elems) {
        features.current().resize(head_elems);
    }

    #pragma omp parallel
    {
        std::mt19937_64 rng(123456789ULL + static_cast<unsigned long long>(omp_get_thread_num()));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        #pragma omp for schedule(static)
        for (int node = 0; node < n_nodes[0]; ++node) {
            double* base = features.current().data() + static_cast<std::size_t>(node) * dims;
            for (int d = 0; d < r; ++d) {
                double val = dist(rng);
                base[d] = val;
                base[d + r] = -val;
            }
        }
    }

    double result_sum = 0.0;
    for (int layer = 0; layer < layers; ++layer) {
        std::vector<double>& curr = features.current();
        const auto& src_idx = index_list[layer].first;
        const auto& dst_idx = index_list[layer].second;
        if (src_idx.size() != dst_idx.size()) {
            throw std::runtime_error("Source and destination index arrays must match in length.");
        }
        const int dst_nodes = n_nodes[layer + 1];
        build_adjacency(src_idx, dst_idx, dst_nodes, workspace);
        index_list[layer].first.clear();
        index_list[layer].second.clear();

        std::vector<double>& next = features.next();
        const std::size_t next_elems = static_cast<std::size_t>(dst_nodes) * dims;
        if (next.size() < next_elems) {
            next.resize(next_elems);
        }
        std::fill(next.begin(), next.begin() + next_elems, kNegInf);

        #pragma omp parallel for schedule(dynamic, 64)
        for (int dst = 0; dst < dst_nodes; ++dst) {
            double* dst_ptr = next.data() + static_cast<std::size_t>(dst) * dims;
            for (int pos = workspace.offsets[dst]; pos < workspace.offsets[dst + 1]; ++pos) {
                const int src = workspace.adjacency[pos];
                if (src < 0 || src >= n_nodes[layer]) {
                    throw std::runtime_error("Source index out of bounds.");
                }
                const double* src_ptr = curr.data() + static_cast<std::size_t>(src) * dims;
                for (int d = 0; d < dims; ++d) {
                    dst_ptr[d] = std::max(dst_ptr[d], src_ptr[d]);
                }
            }
        }

        if (layer == layers - 1) {
            #pragma omp parallel for reduction(+:result_sum)
            for (int node = 0; node < dst_nodes; ++node) {
                const double* ptr = next.data() + static_cast<std::size_t>(node) * dims;
                double tmp = 0.0;
                for (int d = 0; d < dims; ++d) {
                    tmp += ptr[d];
                }
                const double value = 2.0 / (1.0 - (tmp / r)) - 1.0;
                result_sum += value;
            }
        } else {
            features.rotate();
        }
    }

    return result_sum;
}

py::object CRange_pipeline_bidir(int N, py::list input_list, int r = 16,
                                 bool cali = false, bool return_core_time = false) {
    (void)N;
    if (cali) {
        throw std::runtime_error("Calibration mode is not supported in the pipeline implementation.");
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> index_list = convert_to_vector_pair(input_list);
    std::vector<int> n_nodes;
    double result = 0.0;
    double core_time = 0.0;

    {
        py::gil_scoped_release release;
        const auto start = std::chrono::high_resolution_clock::now();
        cool_filter_bidirectional(index_list);
        n_nodes = remap_chain(index_list);
        result = compute_pipeline(index_list, n_nodes, r);
        const auto end = std::chrono::high_resolution_clock::now();
        core_time = std::chrono::duration<double>(end - start).count();
    }

    if (return_core_time) {
        return py::make_tuple(result, core_time);
    }
    return py::float_(result);
}

py::object extract_coremat(int N, py::list input_list, bool return_core_time = false) {
    (void)N;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> index_list =
        convert_to_vector_pair(input_list);
    double core_time = 0.0;

    {
        py::gil_scoped_release release;
        const auto start = std::chrono::high_resolution_clock::now();
        cool_filter_bidirectional(index_list);
        const auto end = std::chrono::high_resolution_clock::now();
        core_time = std::chrono::duration<double>(end - start).count();
    }

    py::list out;
    for (auto& p : index_list) {
        const auto n_left = static_cast<py::ssize_t>(p.first.size());
        const auto n_right = static_cast<py::ssize_t>(p.second.size());
        py::array_t<int> r(n_left);
        py::array_t<int> c(n_right);
        if (n_left > 0) {
            std::memcpy(r.mutable_data(), p.first.data(),
                        static_cast<std::size_t>(n_left) * sizeof(int));
        }
        if (n_right > 0) {
            std::memcpy(c.mutable_data(), p.second.data(),
                        static_cast<std::size_t>(n_right) * sizeof(int));
        }
        out.append(py::make_tuple(std::move(r), std::move(c)));
    }

    if (return_core_time) {
        return py::make_tuple(out, core_time);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(Ranger13, m) {
    m.doc() = "Ranger13: Ranger12 with bitmap-native coolfull pruning.";
    m.def("CRange", &CRange_pipeline_bidir,
          "Compute the CRange statistic using the Ranger13 backend "
          "(bitmap-native coolfull pruning with AVX-512 compaction).",
          py::arg("N"), py::arg("index_list"), py::arg("r") = 16,
          py::arg("cali") = false, py::arg("return_core_time") = false);
    m.def("extract_coremat", &extract_coremat,
          "Run the coolfull-equivalent bidirectional cool_filter (CoreMat) "
          "sparsification with bitmap-native indices + AVX-512 stream-compaction; "
          "return the pruned (row, col) numpy.int32 arrays per layer.",
          py::arg("N"), py::arg("index_list"), py::arg("return_core_time") = false);
}
