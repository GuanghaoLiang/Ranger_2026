#include "smcm_runner_common.hpp"

#include "src/original.hpp"
#include "src/smcm.hpp"
#include "utility.h"

#include "mkl.h"
#include "mkl_spblas.h"

#ifdef HAVE_KOKKOS
#include "src/smcm_kk.hpp"
#include <Kokkos_Core.hpp>
#endif

#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>

namespace {

const char* env_or_default(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? value : fallback;
}

inline bool append_text(const std::string& path, const std::string& text) {
    std::ofstream outfile(path, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << path << std::endl;
        return false;
    }
    outfile << text;
    return true;
}

inline bool append_line(const std::string& path, const std::string& text) {
    return append_text(path, text + "\n");
}

std::pair<CooList, double> build_matcore(CooList& index_list, const ShapeList& shape_list, int p) {
    std::vector<sparse_matrix_t> sparse_matrices(p), sparse_csr(p);
    std::vector<std::pair<std::vector<int>, std::vector<int>>> bool_index_list(p), mat_core(p);
    std::vector<std::set<int>> right_set(p), left_set(p);

#pragma omp parallel for
    for (int i = 0; i < p; ++i) {
        int nnz = static_cast<int>(std::get<0>(index_list[i]).size());
        std::vector<double> values_double(nnz, 1.0);
        if (i == 0) {
            mkl_sparse_d_create_coo(&sparse_matrices[i], SPARSE_INDEX_BASE_ZERO, shape_list[i].first,
                                    shape_list[i].second, nnz, std::get<1>(index_list[i]).data(),
                                    std::get<0>(index_list[i]).data(), values_double.data());
        } else {
            mkl_sparse_d_create_coo(&sparse_matrices[i], SPARSE_INDEX_BASE_ZERO, shape_list[i].second,
                                    shape_list[i].first, nnz, std::get<0>(index_list[i]).data(),
                                    std::get<1>(index_list[i]).data(), values_double.data());
        }

        MKL_INT info = mkl_sparse_convert_csr(sparse_matrices[i], SPARSE_OPERATION_NON_TRANSPOSE, &sparse_csr[i]);
        if (info != 0) {
            std::cerr << "Error converting to CSR format: " << info << std::endl;
        }
        mkl_sparse_destroy(sparse_matrices[i]);
    }

#pragma omp parallel for
    for (int i = 0; i < p; ++i) {
        bool_index_list[i] = std::make_pair(std::get<0>(index_list[i]), std::get<1>(index_list[i]));
        right_set[i] = get_row_csr(sparse_csr[i], shape_list[i].first, shape_list[i].second);
        mkl_sparse_destroy(sparse_csr[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    mat_core = coolfull(bool_index_list, shape_list, right_set, left_set);
    auto end = std::chrono::high_resolution_clock::now();
    double core_elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Core computation time: " << core_elapsed << " seconds" << std::endl;

    CooList matcore_coo(p);
#pragma omp parallel for
    for (int i = 0; i < p; ++i) {
        matcore_coo[i] = std::make_tuple(mat_core[i].first, mat_core[i].second,
                                         std::vector<float>(mat_core[i].first.size(), 1.0f));
    }

    return {matcore_coo, core_elapsed};
}

long run_smcm_once(SmcmBackend backend, const CooList& coo_list, const ShapeList& shape_list,
                   const std::string& par, std::set<std::string>& failed_cases) {
    if (backend == SmcmBackend::CPU) {
        return perform_smcm_coo(coo_list, shape_list, par);
    }

#ifdef HAVE_KOKKOS
    return smcm_coo_kokkos(coo_list, shape_list, failed_cases, par);
#else
    (void)coo_list;
    (void)shape_list;
    (void)par;
    (void)failed_cases;
    throw std::runtime_error("Kokkos backend requested but this binary was built without HAVE_KOKKOS");
#endif
}

bool write_timing(SmcmBackend backend, SmcmPreset preset, const std::string& output_path, long result_nnz,
                  double elapsed) {
    if (backend == SmcmBackend::Kokkos && preset == SmcmPreset::GtKk && result_nnz < 0) {
        return append_text(output_path, "-1 ");
    }
    return append_text(output_path, std::to_string(elapsed) + " ");
}

std::string ensure_trailing_slash(std::string path) {
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    return path;
}

}  // namespace

void apply_env_defaults(SmcmRunConfig& config) {
    if (config.matrix_root.empty()) {
        config.matrix_root = "../materials/input/graph/";
    }
    if (config.metapath_root.empty()) {
        config.metapath_root = "../materials/input/metapath/";
    }
    config.matrix_root = ensure_trailing_slash(
        env_or_default("SMCM_MATRIX_ROOT", config.matrix_root.c_str()));
    config.metapath_root = ensure_trailing_slash(
        env_or_default("SMCM_METAPATH_ROOT", config.metapath_root.c_str()));
    config.output_root = ensure_trailing_slash(
        env_or_default("SMCM_OUTPUT_ROOT", "../result/smcm/"));
    const char* order = std::getenv("SMCM_MULTIPLY_ORDER");
    if (order != nullptr && order[0] != '\0') {
        config.multiply_order = order;
    }
    const char* max_mp = std::getenv("SMCM_MAX_METAPATHS");
    if (max_mp != nullptr && max_mp[0] != '\0') {
        config.max_metapaths = std::stoi(max_mp);
    }
}

bool parse_multiply_order_arg(const std::string& value) {
    return value == "all" || value == "l2r";
}

void parse_optional_order_args(int argc, char* argv[], int start_idx, SmcmRunConfig& config) {
    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--order") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --order");
            }
            const std::string order = argv[++i];
            if (!parse_multiply_order_arg(order)) {
                throw std::runtime_error("Unknown --order value: " + order + " (use all or l2r)");
            }
            config.multiply_order = order;
        }
    }
}

SmcmRunConfig preset_to_config(SmcmPreset preset) {
    SmcmRunConfig config;
    apply_env_defaults(config);
    switch (preset) {
        case SmcmPreset::Gt:
            config.backend = SmcmBackend::CPU;
            config.use_core = false;
            break;
        case SmcmPreset::GtCore:
            config.backend = SmcmBackend::CPU;
            config.use_core = true;
            break;
        case SmcmPreset::GtKk:
            config.backend = SmcmBackend::Kokkos;
            config.use_core = false;
            break;
        case SmcmPreset::GtKkCore:
            config.backend = SmcmBackend::Kokkos;
            config.use_core = true;
            break;
    }
    return config;
}

SmcmOutputPaths resolve_output_paths(const SmcmRunConfig& config) {
    SmcmOutputPaths paths;
    const std::string base = config.output_root + config.dataset + "/";
    const std::string prefix = config.dataset + std::to_string(config.p);

    if (config.use_core) {
        paths.output = base + prefix + "_fullcore.txt";
        paths.log = base + prefix + "_fullcore_log.txt";
    } else if (config.backend == SmcmBackend::Kokkos) {
        paths.output = base + prefix + "kk16.txt";
        paths.log = base + prefix + "_logkk16.txt";
    } else {
        paths.output = base + prefix + ".txt";
        paths.log = base + prefix + "_log.txt";
    }

    return paths;
}

void print_smcm_run_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " <p> <dataset> [--backend cpu|kokkos] [--core|--no-core] [--order all|l2r]\n"
        << "\n"
        << "Environment overrides:\n"
        << "  SMCM_MATRIX_ROOT       default: materials/input/graph/\n"
        << "  SMCM_METAPATH_ROOT     default: materials/input/metapath/\n"
        << "  SMCM_OUTPUT_ROOT       default: ../result/smcm/\n"
        << "  SMCM_MULTIPLY_ORDER    default: all (use l2r for left-to-right chain multiply)\n"
        << "  SMCM_MAX_METAPATHS     optional limit on number of metapaths to run\n"
        << "\n"
        << "Metapath format: compressed p+1 tokens, e.g. 251-387-396-73 for p=3\n"
        << "Matrix files: {src}-{dst}.txt under SMCM_MATRIX_ROOT/{dataset}/\n"
        << "\n"
        << "Examples:\n"
        << "  " << program << " 3 DBPedia --backend cpu --no-core --order l2r\n"
        << "  " << program << " 3 DBPedia --backend cpu --core --order l2r\n";
}

int parse_smcm_run_args(int argc, char* argv[], SmcmRunConfig& config) {
    if (argc < 3) {
        print_smcm_run_usage(argv[0]);
        return 1;
    }

    apply_env_defaults(config);
    config.p = std::stoi(argv[1]);
    config.dataset = argv[2];

    bool core_set = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--backend") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --backend\n";
                return 1;
            }
            const std::string backend = argv[++i];
            if (backend == "cpu") {
                config.backend = SmcmBackend::CPU;
            } else if (backend == "kokkos") {
                config.backend = SmcmBackend::Kokkos;
            } else {
                std::cerr << "Unknown backend: " << backend << "\n";
                return 1;
            }
        } else if (arg == "--core") {
            config.use_core = true;
            core_set = true;
        } else if (arg == "--no-core") {
            config.use_core = false;
            core_set = true;
        } else if (arg == "--order") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --order\n";
                return 1;
            }
            const std::string order = argv[++i];
            if (!parse_multiply_order_arg(order)) {
                std::cerr << "Unknown --order value: " << order << " (use all or l2r)\n";
                return 1;
            }
            config.multiply_order = order;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_smcm_run_usage(argv[0]);
            return 1;
        }
    }

    if (!core_set) {
        config.use_core = false;
    }

    return 0;
}

int smcm_run_with_preset(int argc, char* argv[], SmcmPreset preset) {
    if (argc < 3) {
        print_smcm_run_usage(argv[0]);
        return 1;
    }

    SmcmRunConfig config = preset_to_config(preset);
    config.p = std::stoi(argv[1]);
    config.dataset = argv[2];
    try {
        parse_optional_order_args(argc, argv, 3, config);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_smcm_run_usage(argv[0]);
        return 1;
    }
    return run_smcm_benchmark(config);
}

namespace {

struct MetapathLoadResult {
    std::vector<MetapathJob> jobs;
    std::string source_path;
};

std::string join_path(const std::string& root, const std::string& suffix) {
    if (root.empty()) {
        return suffix;
    }
    if (!suffix.empty() && suffix.front() == '/') {
        return root + suffix.substr(1);
    }
    return root + suffix;
}

MetapathLoadResult load_metapath_jobs(const SmcmRunConfig& config) {
    MetapathLoadResult result;
    const std::vector<std::string> candidates = {
        join_path(config.metapath_root, config.dataset + "/" + config.dataset + "-l" +
                                         std::to_string(config.p) + ".txt"),
        join_path(config.metapath_root, "len_" + std::to_string(config.p) + "/" + config.dataset +
                                         std::to_string(config.p) + ".txt"),
    };

    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        std::ifstream probe(candidate);
        if (!probe.is_open()) {
            continue;
        }
        probe.close();

        result.source_path = candidate;
        result.jobs = read_metapath_jobs(candidate);
        if (config.max_metapaths > 0 &&
            static_cast<int>(result.jobs.size()) > config.max_metapaths) {
            result.jobs.resize(static_cast<size_t>(config.max_metapaths));
        }
        return result;
    }

    throw std::runtime_error(
        "No metapath file found for dataset=" + config.dataset + " p=" + std::to_string(config.p));
}

}  // namespace

int run_smcm_benchmark(const SmcmRunConfig& config) {
    if (config.backend == SmcmBackend::Kokkos) {
#ifndef HAVE_KOKKOS
        std::cerr << "Kokkos backend is not available in this binary.\n";
        return 1;
#endif
    }

    int mp_start = 0;
    int par_start = 0;
    bool resume = false;

    int cat = catalan_number(config.p);
    std::cout << "Catalan number for " << config.p << " is: " << cat << std::endl;
    std::cout << "Backend: " << (config.backend == SmcmBackend::CPU ? "cpu" : "kokkos")
              << ", use_core: " << (config.use_core ? "true" : "false")
              << ", multiply_order: " << config.multiply_order << std::endl;

    std::vector<std::string> parentheses;
    if (config.multiply_order == "l2r") {
        parentheses = {"l2r"};
    } else {
        generate_parentheses(1, config.p, parentheses);
    }

    const std::string filepath = config.matrix_root + config.dataset;
    const SmcmOutputPaths paths = resolve_output_paths(config);

    MetapathLoadResult metapaths;
    try {
        metapaths = load_metapath_jobs(config);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    std::cout << "Metapath source: " << metapaths.source_path << std::endl;
    std::cout << "Loaded metapaths: " << metapaths.jobs.size() << std::endl;

    std::tie(mp_start, par_start) = getFirstAndLastFieldOfLastLine(paths.log, resume);
    std::cout << "mp_start: " << mp_start << ", par_start: " << par_start << ", resume " << resume << std::endl;

#ifdef HAVE_KOKKOS
    struct KokkosGuard {
        explicit KokkosGuard(SmcmBackend backend) : active(backend == SmcmBackend::Kokkos) {
            if (active) {
                Kokkos::initialize();
            }
        }
        ~KokkosGuard() {
            if (active) {
                Kokkos::finalize();
            }
        }
        bool active;
    } kokkos_guard(config.backend);
#endif

    SmcmPreset preset = SmcmPreset::Gt;
    if (config.backend == SmcmBackend::Kokkos) {
        preset = config.use_core ? SmcmPreset::GtKkCore : SmcmPreset::GtKk;
    } else {
        preset = config.use_core ? SmcmPreset::GtCore : SmcmPreset::Gt;
    }

    for (int mp_idx = mp_start; mp_idx < static_cast<int>(metapaths.jobs.size()); ++mp_idx) {
        const MetapathJob& job = metapaths.jobs[mp_idx];

        CooList index_list;
        ShapeList shape_list;
        std::map<std::string, double> myDict;
        std::set<std::string> failed_cases;
        long result_nnz = 0;
        double max_time = 0.0;
        double min_time = std::numeric_limits<double>::max();
        double core_elapsed = 0.0;

        try {
            std::tie(index_list, shape_list) = read_mat_by_mp<int, float>(job.mp, filepath, config.p);
        } catch (const std::exception& e) {
            std::cerr << "Error processing metapath " << job.mp << ": " << e.what() << std::endl;
            continue;
        }

        std::cout << "Computing MP: " << job.mp << std::endl;

        CooList run_input = index_list;
        if (config.use_core) {
            std::tie(run_input, core_elapsed) = build_matcore(index_list, shape_list, config.p);
        }

        if (!resume) {
            if (config.use_core) {
                if (!append_text(paths.output, job.mp + " " + std::to_string(core_elapsed) + " ")) {
                    continue;
                }
            } else if (!append_text(paths.output, job.mp + " ")) {
                continue;
            }
        }

        if (!resume) {
            if (!append_text(paths.log, std::to_string(mp_idx) + "  " + job.mp + " ")) {
                std::cerr << "Failed to open output file: " << paths.log << std::endl;
            }
        } else {
            par_start += 1;
        }

        if (par_start >= static_cast<int>(parentheses.size())) {
            par_start = 0;
            append_line(paths.output, "   ");
            append_line(paths.log, "   ");
            resume = false;
            continue;
        }

        std::cout << par_start << std::endl;

        for (int j = par_start; j < static_cast<int>(parentheses.size()); ++j) {
            const std::string& par = parentheses[j];
            append_text(paths.log, " " + std::to_string(j) + " ");
            append_text(paths.output, "   " + par + ":");

            auto t0 = std::chrono::high_resolution_clock::now();
            result_nnz = run_smcm_once(config.backend, run_input, shape_list, par, failed_cases);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::cout << "Result nnz: " << result_nnz << std::endl;
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;

            myDict[par] = elapsed;
            write_timing(config.backend, preset, paths.output, result_nnz, elapsed);

            max_time = std::max(max_time, elapsed);
            min_time = std::min(min_time, elapsed);
        }

        append_line(paths.log, "  ");
        par_start = 0;
        resume = false;

        double range = max_time - min_time;
        std::string summary = formatEntry(job.mp, range, myDict);
        std::cout << summary << std::endl;
        append_line(paths.output, "  ");
    }

    return 0;
}
