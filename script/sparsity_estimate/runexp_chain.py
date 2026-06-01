import argparse
import os
import re
import sys
import time
import warnings

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore")

_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_IMDB_METAPATH_ROOT = os.path.join(_PKG_ROOT, "materials", "input", "metapath", "IMDB")
_IMDB_GRAPH_ROOT = os.path.join(_PKG_ROOT, "materials", "input", "graph", "IMDB")
for _p in (
    _PKG_ROOT,
    os.path.join(_PKG_ROOT, "baseline"),
    os.path.join(_PKG_ROOT, "script", "sparsity_estimate", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import DensityMap
import LG_EXP1
import MetaAC
import mnc
from Ranger13 import CRange as crange13_cpp
from utils.utils import read_metapath_file, resume, seed_everything


def csr_bool(rows, cols, n):
    rs = np.asarray(rows, dtype=np.int64)
    cs = np.asarray(cols, dtype=np.int64)
    if rs.size == 0:
        return csc_matrix((n, n), dtype=np.uint8).tocsr()
    data = np.ones(rs.size, dtype=np.uint8)
    return coo_matrix((data, (rs, cs)), shape=(n, n), dtype=np.uint8).tocsr()


def bool_mul_support(a, b):
    c = a @ b
    if c.nnz:
        c = c.tocoo()
        c.data = np.ones(len(c.data), dtype=np.uint8)
        c = c.tocsr()
    return c


def ground_truth_nnz_scipy(n, index_list):
    mats = [csr_bool(row, col, n) for row, col in index_list]
    if not mats:
        return 0
    acc = mats[0]
    for mat in mats[1:]:
        acc = bool_mul_support(acc, mat)
    return int(acc.nnz)


def parse():
    parser = argparse.ArgumentParser(
        description="Run sparsity estimators on IMDB metapath chains (GT via SciPy boolean multiply)."
    )
    parser.add_argument("--lengths", type=str, default="3")
    parser.add_argument(
        "--metapath",
        type=str,
        default=_IMDB_METAPATH_ROOT,
        help="Directory containing IMDB-l{L}.txt metapath files",
    )
    parser.add_argument(
        "--adj_path",
        type=str,
        default=_IMDB_GRAPH_ROOT,
        help="Directory containing IMDB adjacency matrices ({id}.txt files)",
    )
    parser.add_argument("--b", type=int, default=256, help="DensityMap block size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        type=str,
        default="MNC,DensityMap,LayerGraphEXP16,MetaAC",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join(_PKG_ROOT, "experiment/exp_chain"),
    )
    parser.add_argument("--max_metapaths", type=int, default=0, help="If >0, only run first K metapaths")
    return parser.parse_args()


def parse_range_method(name):
    match = re.fullmatch(r"(Ranger13)_(\d+)", name)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Unrecognized Range method format: {name}")


def run_crange(N, index_list, r, name, backend):
    result, core_time = backend(N, index_list, r, False, True)
    print(f"-------------Core time for {name}: {core_time:.4f} seconds")
    return result, core_time


def get_function(N, name, index_list, args):
    if name.startswith("Ranger13_"):
        _, r = parse_range_method(name)
        return run_crange(N, index_list, r, name, crange13_cpp)
    if name.startswith("LayerGraphEXP"):
        r = int(name.replace("LayerGraphEXP", ""))
        result = LG_EXP1.layer_graph_chain(N, index_list, r)
        return result[-1] * N * N, 0
    if name == "MNC":
        result = mnc.mnc_chain(N, index_list)
        return result[-1], 0
    if name == "DensityMap":
        result = DensityMap.dm_chain(N, index_list, args.b)
        return result[-1], 0
    if name == "MetaAC":
        result = MetaAC.MetaAC(N, index_list)
        return result[-1] * N * N, 0
    raise ValueError(f"Function {name} not found")


def read_adj_csr(path, mp, j):
    """Load IMDB adjacency matrix mp[2*j+1].txt (3-line COO/CSR format)."""
    csr_path = os.path.join(path, f"{mp[2 * j + 1]}.txt")
    if not os.path.exists(csr_path):
        raise FileNotFoundError(f"Adjacency matrix not found: {csr_path}")

    print(f"Reading adjacency matrix (numpy) from {csr_path}")
    with open(csr_path, "r") as file:
        lines = file.readlines()
    n = int(lines[0].strip())
    row_vals = np.array([int(x) for x in lines[1].split()], dtype=np.int64)
    col_vals = np.array([int(x) for x in lines[2].split()], dtype=np.int64)
    nnz = col_vals.size

    if row_vals.size == n + 1 and row_vals[-1] == nnz:
        counts = np.diff(row_vals)
        if np.any(counts < 0):
            raise ValueError(f"CSR row pointer not monotonic in {csr_path}")
        row = np.repeat(np.arange(n, dtype=np.int64), counts)
    else:
        if row_vals.size != nnz:
            raise ValueError(
                f"Row list length {row_vals.size} mismatch with nnz {nnz} in {csr_path}"
            )
        row = row_vals

    print(f"Finished reading adjacency matrix from {csr_path} (nnz={nnz})")
    return n, row.astype(np.int32), col_vals.astype(np.int32)


def load_index_list(adj_root, metapath, length):
    index_list = []
    n_nodes = 0
    for j in range(length):
        n, row, col = read_adj_csr(adj_root, metapath, j)
        n_nodes = max(n_nodes, n)
        index_list.append([row, col])
    return n_nodes, index_list


def metapath_file_for_length(metapath_root, length):
    candidates = [
        os.path.join(metapath_root, f"IMDB-l{length}.txt"),
        os.path.join(
            _PKG_ROOT,
            "materials",
            "output",
            "accuracy",
            "IMDB",
            f"dm_acc-l{length}.txt",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No IMDB metapath file found for length {length}; tried {candidates}"
    )


def main():
    args = parse()
    seed_everything(args.seed)
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    flag = False

    for length in lengths:
        results = {method: [] for method in methods}

        print(f"Running IMDB experiment with length {length}")
        base_dir = os.path.join(args.output_root, f"len_{length}")
        dataset_dir = os.path.join(base_dir, "IMDB")
        os.makedirs(dataset_dir, exist_ok=True)

        metapath_path = metapath_file_for_length(args.metapath, length)
        print(f"Using metapath file: {metapath_path}")
        print(f"Using adjacency root: {args.adj_path}")

        metapath = read_metapath_file(metapath_path, gt=False)
        if args.max_metapaths > 0:
            metapath = metapath[: args.max_metapaths]
        print(f"Total {len(metapath)} metapaths.")

        existing_results = set()
        if flag and methods:
            resume_path = os.path.join(dataset_dir, f"{methods[0]}.txt")
            existing_results = resume(resume_path)

        for i in tqdm(range(len(metapath))):
            try:
                if flag and metapath[i] in existing_results:
                    continue
                print(f"Processing metapath: {metapath[i]}")
                N, index_list = load_index_list(args.adj_path, metapath[i], length)
                gt_nnz = ground_truth_nnz_scipy(N, index_list)
                if gt_nnz <= 0:
                    continue

                for method in methods:
                    time_start = time.time()
                    esti_nnz, prepare_time = get_function(N, method, index_list, args)
                    time_end = time.time()
                    esti_time = time_end - time_start
                    if esti_nnz <= 0:
                        relative_error = -1
                    else:
                        relative_error = max(esti_nnz, gt_nnz) / min(esti_nnz, gt_nnz)
                    results[method].append(
                        [metapath[i], relative_error, esti_time, prepare_time]
                    )

                for method in methods:
                    out_path = os.path.join(dataset_dir, f"{method}.txt")
                    with open(out_path, "w") as f:
                        for item in results[method]:
                            f.write(f"{item}\n")
            except Exception as e:
                print(f"Error processing metapath {metapath[i]}: {e}")
                continue


if __name__ == "__main__":
    main()
