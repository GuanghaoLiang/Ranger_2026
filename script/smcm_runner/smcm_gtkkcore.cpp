#include "smcm_runner_common.hpp"

int main(int argc, char* argv[]) {
    return smcm_run_with_preset(argc, argv, SmcmPreset::GtKkCore);
}
