#!/usr/bin/env python3
"""Launch unified SMCM benchmark runs from the self-contained smcm_runner folder."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

RUNNER_ROOT = Path(__file__).resolve().parent
_PKG_ROOT = RUNNER_ROOT.parents[1]
DEFAULT_BINARY = RUNNER_ROOT / "smcm_run"
BUILD_SCRIPT = RUNNER_ROOT / "build.sh"

MKL_ROOT = Path(os.environ.get("MKL_ROOT", "/home/guanghao/intel/oneapi/mkl/2025.2"))
KOKKOS_ROOT = Path(os.environ.get("KOKKOS_ROOT", "/home/guanghao/kokkos-4.6.01/kokkos-install"))
KOKKOS_KERNELS_ROOT = Path(
    os.environ.get("KOKKOS_KERNELS_ROOT", "/home/guanghao/kokkos-kernels-4.6.01/install")
)

MODES = {
    "gt": ("cpu", False),
    "gtcore": ("cpu", True),
    "gtkk": ("kokkos", False),
    "gtkkcore": ("kokkos", True),
}

ELAPSED_RE = re.compile(r"^Elapsed time:\s*([0-9.eE+-]+)\s*seconds?\s*$")
CORE_PREP_RE = re.compile(r"^Core computation time:\s*([0-9.eE+-]+)\s*seconds?\s*$")
MP_RE = re.compile(r"^Computing MP:\s*(.+)\s*$")


@dataclass
class MetapathTiming:
    mp: str
    mul_times: list[float] = field(default_factory=list)
    core_prep_time: float | None = None

    @property
    def total_mul_time(self) -> float:
        return float(sum(self.mul_times))


def runner_env(
    output_root: str | None = None,
    metapath_root: str | None = None,
    max_metapaths: int = 0,
) -> dict[str, str]:
    env = os.environ.copy()
    if metapath_root:
        env["SMCM_METAPATH_ROOT"] = metapath_root if metapath_root.endswith("/") else metapath_root + "/"
    elif "SMCM_METAPATH_ROOT" not in env:
        default_metapath = _PKG_ROOT / "materials" / "input" / "metapath"
        if default_metapath.is_dir():
            env["SMCM_METAPATH_ROOT"] = str(default_metapath.resolve()) + "/"
    if output_root is not None:
        env["SMCM_OUTPUT_ROOT"] = output_root
    elif "SMCM_OUTPUT_ROOT" not in env:
        env["SMCM_OUTPUT_ROOT"] = str((_PKG_ROOT / "experiment" / "smcm").resolve()) + "/"

    if "SMCM_MATRIX_ROOT" not in env:
        env["SMCM_MATRIX_ROOT"] = str((_PKG_ROOT / "materials" / "input" / "graph").resolve()) + "/"

    if max_metapaths > 0:
        env["SMCM_MAX_METAPATHS"] = str(max_metapaths)

    lib_paths = [
        str(MKL_ROOT / "lib" / "intel64"),
        str(KOKKOS_ROOT / "lib"),
        str(KOKKOS_KERNELS_ROOT / "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(p for p in lib_paths if p)
    return env


def resolve_backend_and_core(args: argparse.Namespace) -> tuple[str, bool]:
    if args.mode:
        return MODES[args.mode]
    return args.backend, args.core


def build_command(
    binary: Path,
    p: int,
    dataset: str,
    backend: str,
    use_core: bool,
    multiply_order: str = "all",
) -> list[str]:
    name = binary.name
    if name in {"smcm_gt", "smcm_gtcore", "smcm_gtkk", "smcm_gtkkcore"}:
        cmd = [str(binary), str(p), dataset]
        if multiply_order != "all":
            cmd.extend(["--order", multiply_order])
        return cmd
    cmd = [
        str(binary),
        str(p),
        dataset,
        "--backend",
        backend,
        "--core" if use_core else "--no-core",
    ]
    if multiply_order != "all":
        cmd.extend(["--order", multiply_order])
    return cmd


def compare_binaries(args: argparse.Namespace, backend: str) -> tuple[Path, Path]:
    if args.compare_no_core_bin and args.compare_core_bin:
        return args.compare_no_core_bin, args.compare_core_bin

    unified = args.binary
    if unified.is_file() and unified.name == "smcm_run":
        return unified, unified

    if backend == "kokkos":
        return RUNNER_ROOT / "smcm_gtkk", RUNNER_ROOT / "smcm_gtkkcore"
    return RUNNER_ROOT / "smcm_gt", RUNNER_ROOT / "smcm_gtcore"


def parse_benchmark_timings(stdout: str) -> list[MetapathTiming]:
    entries: list[MetapathTiming] = []
    current: MetapathTiming | None = None

    for line in stdout.splitlines():
        mp_match = MP_RE.match(line)
        if mp_match:
            current = MetapathTiming(mp=mp_match.group(1).strip())
            entries.append(current)
            continue

        if current is None:
            continue

        core_match = CORE_PREP_RE.match(line)
        if core_match:
            current.core_prep_time = float(core_match.group(1))
            continue

        elapsed_match = ELAPSED_RE.match(line)
        if elapsed_match:
            current.mul_times.append(float(elapsed_match.group(1)))

    return entries


def run_benchmark(
    cmd: list[str],
    env: dict[str, str],
) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        check=False,
        cwd=RUNNER_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def print_compare_table(
    no_core: list[MetapathTiming],
    with_core: list[MetapathTiming],
) -> list[dict]:
    core_by_mp = {item.mp: item for item in with_core}
    rows: list[dict] = []

    print("\n=== Core vs No-Core timing ===")
    print(
        f"{'metapath':<28} {'no_core(s)':>11} {'core_prep':>10} {'core_mul':>10} "
        f"{'core_e2e':>10} {'mul_spd':>8} {'e2e_spd':>8}"
    )
    print("-" * 98)

    for item in no_core:
        core_item = core_by_mp.get(item.mp)
        if core_item is None:
            print(f"{item.mp:<28} {'MISSING':>11}")
            continue

        no_core_t = item.total_mul_time
        core_prep_t = core_item.core_prep_time or 0.0
        core_mul_t = core_item.total_mul_time
        core_e2e_t = core_prep_t + core_mul_t
        mul_speedup = no_core_t / core_mul_t if core_mul_t > 0 else float("inf")
        e2e_speedup = no_core_t / core_e2e_t if core_e2e_t > 0 else float("inf")

        row = {
            "metapath": item.mp,
            "no_core_mul_sec": round(no_core_t, 6),
            "core_prep_sec": round(core_prep_t, 6),
            "with_core_mul_sec": round(core_mul_t, 6),
            "with_core_e2e_sec": round(core_e2e_t, 6),
            "mul_speedup": round(mul_speedup, 4),
            "e2e_speedup": round(e2e_speedup, 4),
            "speedup": round(mul_speedup, 4),
        }
        rows.append(row)
        print(
            f"{item.mp:<28} {no_core_t:11.6f} {core_prep_t:10.6f} {core_mul_t:10.6f} "
            f"{core_e2e_t:10.6f} {mul_speedup:8.4f} {e2e_speedup:8.4f}"
        )

    if rows:
        avg_no = sum(r["no_core_mul_sec"] for r in rows) / len(rows)
        avg_prep = sum(r["core_prep_sec"] for r in rows) / len(rows)
        avg_core = sum(r["with_core_mul_sec"] for r in rows) / len(rows)
        avg_e2e = sum(r["with_core_e2e_sec"] for r in rows) / len(rows)
        avg_mul_speedup = avg_no / avg_core if avg_core > 0 else float("inf")
        avg_e2e_speedup = avg_no / avg_e2e if avg_e2e > 0 else float("inf")
        print("-" * 98)
        print(
            f"{'AVERAGE':<28} {avg_no:11.6f} {avg_prep:10.6f} {avg_core:10.6f} "
            f"{avg_e2e:10.6f} {avg_mul_speedup:8.4f} {avg_e2e_speedup:8.4f}"
        )

    return rows


def run_compare_core(args: argparse.Namespace) -> int:
    backend, _ = resolve_backend_and_core(args)
    no_core_bin, core_bin = compare_binaries(args, backend)

    for binary in (no_core_bin, core_bin):
        if not binary.is_file():
            print(f"Binary not found: {binary}", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory(prefix="smcm_compare_") as tmp:
        out_no_core = str(Path(tmp) / "no_core") + "/"
        out_with_core = str(Path(tmp) / "with_core") + "/"
        Path(out_no_core, args.dataset).mkdir(parents=True, exist_ok=True)
        Path(out_with_core, args.dataset).mkdir(parents=True, exist_ok=True)

        cmd_no = build_command(
            no_core_bin, args.p, args.dataset, backend, use_core=False, multiply_order=args.order
        )
        cmd_core = build_command(
            core_bin, args.p, args.dataset, backend, use_core=True, multiply_order=args.order
        )

        metapath_root = args.metapath_root or None
        print("Working directory:", RUNNER_ROOT)
        print("Backend:", backend)
        print("\n[1/2] Run without core")
        print("Command:", " ".join(cmd_no))
        env_no = runner_env(out_no_core, metapath_root, args.max_metapaths)
        rc_no, out_no, _ = run_benchmark(cmd_no, env_no)
        if rc_no != 0:
            return rc_no

        print("\n[2/2] Run with core")
        print("Command:", " ".join(cmd_core))
        env_core = runner_env(out_with_core, metapath_root, args.max_metapaths)
        rc_core, out_core, _ = run_benchmark(cmd_core, env_core)
        if rc_core != 0:
            return rc_core

        rows = print_compare_table(
            parse_benchmark_timings(out_no),
            parse_benchmark_timings(out_core),
        )

        if args.compare_json:
            payload = {
                "dataset": args.dataset,
                "p": args.p,
                "backend": backend,
                "rows": rows,
            }
            out_path = Path(args.compare_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"\nWrote comparison JSON to {out_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MatCore SMCM benchmark suite.")
    parser.add_argument("p", type=int, help="chain length")
    parser.add_argument("dataset", help="dataset name, e.g. IMDB")
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        help="legacy preset matching smcm_gt / smcm_gtcore / smcm_gtkk / smcm_gtkkcore",
    )
    parser.add_argument("--backend", choices=["cpu", "kokkos"], default="cpu")
    parser.add_argument("--core", action="store_true", help="apply MatCore coolfull preprocessing")
    parser.add_argument(
        "--order",
        choices=["all", "l2r"],
        default="all",
        help="multiplication order: all parenthesis variants (default) or l2r (left-to-right)",
    )
    parser.add_argument(
        "--compare-core",
        action="store_true",
        help="run once without core and once with core, then compare mul/e2e timing",
    )
    parser.add_argument(
        "--metapath-root",
        type=str,
        default="",
        help="override SMCM_METAPATH_ROOT (default: materials/input/metapath)",
    )
    parser.add_argument(
        "--max-metapaths",
        type=int,
        default=0,
        help="if >0, only run the first K metapaths from the file",
    )
    parser.add_argument(
        "--compare-json",
        type=str,
        default="",
        help="optional path to write core/no-core comparison as JSON (with --compare-core)",
    )
    parser.add_argument(
        "--compare-no-core-bin",
        type=Path,
        default=None,
        help="binary for no-core run in compare mode (default: smcm_gt or smcm_run)",
    )
    parser.add_argument(
        "--compare-core-bin",
        type=Path,
        default=None,
        help="binary for core run in compare mode (default: smcm_gtcore or smcm_run)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"path to smcm_run binary (default: {DEFAULT_BINARY})",
    )
    parser.add_argument("--build", action="store_true", help="build binaries before running")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode and (args.backend != "cpu" or args.core):
        parser.error("use either --mode or explicit --backend/--core, not both")
    if args.compare_core and args.mode:
        parser.error("use either --compare-core or --mode, not both")
    if args.compare_core and args.core:
        parser.error("use either --compare-core or --core, not both")

    if args.build:
        if not BUILD_SCRIPT.is_file():
            parser.error(f"build script not found: {BUILD_SCRIPT}")
        build_result = subprocess.run(["bash", str(BUILD_SCRIPT)], cwd=RUNNER_ROOT, check=False)
        if build_result.returncode != 0:
            return build_result.returncode

    if args.compare_core:
        return run_compare_core(args)

    backend, use_core = resolve_backend_and_core(args)
    cmd = build_command(args.binary, args.p, args.dataset, backend, use_core, args.order)
    metapath_root = args.metapath_root or None
    env = runner_env(metapath_root=metapath_root, max_metapaths=args.max_metapaths)
    print("Working directory:", RUNNER_ROOT)
    print("Command:", " ".join(cmd))
    if args.dry_run:
        print("SMCM_METAPATH_ROOT:", env.get("SMCM_METAPATH_ROOT", ""))
        print("SMCM_OUTPUT_ROOT:", env["SMCM_OUTPUT_ROOT"])
        print("SMCM_MATRIX_ROOT:", env["SMCM_MATRIX_ROOT"])
        print("SMCM_MAX_METAPATHS:", env.get("SMCM_MAX_METAPATHS", "0"))
        return 0

    return subprocess.run(cmd, check=False, cwd=RUNNER_ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
