#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MKL_ROOT="${MKL_ROOT:-/home/guanghao/intel/oneapi/mkl/2025.2}"
KOKKOS_ROOT="${KOKKOS_ROOT:-/home/guanghao/kokkos-4.6.01/kokkos-install}"
KOKKOS_KERNELS_ROOT="${KOKKOS_KERNELS_ROOT:-/home/guanghao/kokkos-kernels-4.6.01/install}"

COMMON_CXXFLAGS=(
  -O3
  -std=c++17
  -fopenmp
  -I"$ROOT"
  -I"$KOKKOS_ROOT/include"
  -I"$KOKKOS_KERNELS_ROOT/include"
  -I"$MKL_ROOT/include"
)

COMMON_LIBS=(
  -L"$MKL_ROOT/lib/intel64"
  -lmkl_intel_lp64
  -lmkl_sequential
  -lmkl_core
  -L"$KOKKOS_KERNELS_ROOT/lib"
  -lkokkoskernels
  -L"$KOKKOS_ROOT/lib"
  -lkokkoscore
  -lkokkoscontainers
  -lkokkossimd
  -lpthread
  -lm
  -ldl
)

KOKKOS_CXXFLAGS=(
  -DHAVE_KOKKOS
)

build_cpu() {
  local target="$1"
  local source="$2"
  echo "Building $target (CPU preset)..."
  g++ "${COMMON_CXXFLAGS[@]}" "$source" smcm_runner_common.cpp -o "$target" "${COMMON_LIBS[@]}"
}

build_kokkos() {
  local target="$1"
  local source="$2"
  echo "Building $target (Kokkos preset)..."
  g++ "${COMMON_CXXFLAGS[@]}" "${KOKKOS_CXXFLAGS[@]}" "$source" smcm_runner_common.cpp -o "$target" \
    "${COMMON_LIBS[@]}"
}

build_cpu smcm_gt smcm_gt.cpp
build_cpu smcm_gtcore smcm_gtcore.cpp
build_kokkos smcm_gtkk smcm_gtkk.cpp
build_kokkos smcm_gtkkcore smcm_gtkkcore.cpp
build_kokkos smcm_run smcm_run.cpp

echo "Done. Built in $ROOT:"
echo "  smcm_gt smcm_gtcore smcm_gtkk smcm_gtkkcore smcm_run"
echo "Tip: export LD_LIBRARY_PATH=\"$MKL_ROOT/lib/intel64:$KOKKOS_ROOT/lib:$KOKKOS_KERNELS_ROOT/lib:\$LD_LIBRARY_PATH\""
