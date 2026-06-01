#include "smcm_runner_common.hpp"

int main(int argc, char* argv[]) {
    SmcmRunConfig config;
    if (parse_smcm_run_args(argc, argv, config) != 0) {
        return 1;
    }
    return run_smcm_benchmark(config);
}
