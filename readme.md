This is the official implementation of the paper "Structure Sparsification and Sparsity Estimation for Efficient Sparse Matrix Multiplication"

## SMCM core comparison

This section describes how to reproduce the **with-core vs without-core** matrix-chain multiplication experiment.

For each metapath (a chain of 3 sparse matrix multiplications), the runner compares:

| Setting | Steps timed |
|---------|-------------|
| **no-core** | SMCM chain multiply on original matrices |
| **with-core** | core pruning (`coolfull`) + SMCM chain multiply on pruned matrices |

The comparison script reports per-metapath and average:

- `no_core` — multiplication time without pruning
- `core_prep` — core pruning time
- `core_mul` — multiplication time after pruning
- `core_e2e` — `core_prep + core_mul` (end-to-end with core)
- `mul_spd` — `no_core / core_mul`
- `e2e_spd` — `no_core / core_e2e`

### Bundled inputs

**Metapaths** 

- example (`materials/input/metapath/DBPedia/DBPedia-l3.txt`)

**Adjacency matrices** 

- One file per compressed edge type pair: `{src}-{dst}.txt` (e.g. `251-387.txt`).
- example (`materials/input/graph/DBPedia/`)


### Dependencies

Build and run `script/smcm_runner/` require:

| Component | Purpose |
|-----------|---------|
| **g++** (C++17) + **OpenMP** (`-fopenmp`) | compile SMCM runner |
| **Intel MKL** | `--backend cpu` multiplication (`mkl_sparse_spmm`) and core pruning |
| **Kokkos** + **Kokkos Kernels** | `--backend kokkos` multiplication; also linked for unified binary |
| **Python 3** | launch script `run_smcm.py` |

Set these environment variables before building (adjust paths to your installation):

```bash
export MKL_ROOT=/path/to/mkl         
export KOKKOS_ROOT=/path/to/kokkos-install
export KOKKOS_KERNELS_ROOT=/path/to/kokkos-kernels-install
export LD_LIBRARY_PATH="$MKL_ROOT/lib/intel64:$KOKKOS_ROOT/lib:$KOKKOS_KERNELS_ROOT/lib:$LD_LIBRARY_PATH"
```

The build script defaults are in `script/smcm_runner/build.sh`.

### Build

From the repository root:

```bash
bash VLDB2026/script/smcm_runner/build.sh
```


```bash
python3 run_smcm.py 3 DBPedia \
  --compare-core \
  --backend cpu \
  --order l2r \
  --compare-json /tmp/dbpedia_core_compare_l3.json
```
As the full DBPedia/FreeBase dataset consumes large memory, we provide 20 example metapaths with the matrices involved. 

## Sparsity estimation on graph metapaths (IMDB)

This section describes how to reproduce the **graph metapath sparsity-estimation** experiment with `script/sparsity_estimate/runexp_chain.py`, using only scripts and data under `VLDB2026/`. The bundled example uses **IMDB** with chain length **3**.

### What the experiment does

For each metapath (an ordered chain of sparse matrix multiplications), the runner:

1. Loads the adjacency matrices along the metapath.
2. Computes **ground-truth nnz** by SciPy boolean chain multiplication (left-to-right).
3. Runs one or more sparsity estimators and records, per metapath and method:
   - relative error `max(est, gt) / min(est, gt)`
   - wall-clock time
   - predictor core/prepare time (if applicable)

Default methods: **MNC**, **DensityMap**, **LayerGraphEXP16**, **MetaAC**. You can also pass **Ranger13_16**.

Metapath files need **only the path tokens** (one line per metapath).

### Bundled inputs (IMDB)


| Component | Purpose |
|-----------|---------|
| **Python 3** | launch script |
| **NumPy**, **SciPy**, **tqdm** | matrix I/O, SciPy GT chain multiply, progress bar |
| **pybind11** + **g++** (C++11/17, **OpenMP**) | compile predictor extensions if `.so` files are missing |


```bash
cd VLDB2026

python3 script/sparsity_estimate/runexp_chain.py \
  --lengths 3 \
  --methods MNC,DensityMap,LayerGraphEXP16,MetaAC \
  --output_root experiment/exp_chain
```

### Example: Run on all bundled IMDB len-3 metapaths

```bash
python3 script/sparsity_estimate/runexp_chain.py \
  --lengths 3 \
  --output_root experiment/exp_chain
```

Metapath file resolution order (first existing path wins):

1. `{metapath}/IMDB-l{L}.txt`
2. `materials/output/accuracy/IMDB/dm_acc-l{L}.txt`



## Sparsity estimation on SuiteSparse chains

This section describes how to reproduce the **SuiteSparse matrix-chain sparsity-estimation** experiment with `script/sparsity_estimate/run_suite_chain_predictors.py`. Readers should first download the matrices listed in `matrices_catalog.json` (see below).

### What the experiment does

For each ordered matrix chain listed in `combinations.jsonl`, the runner:

1. Loads input matrices from `materials/input/SuiteSparse/matrix/` via `matrices_catalog.json`.
2. Computes **ground-truth nnz** by SciPy boolean chain multiplication on the upper-triangle support (common dimension `n = min(shape)` across the chain).
3. Runs predictors in **isolated subprocesses** (default timeout 300 s per method) and writes one JSONL row per combination:

```json
{"combination": ["GL7d17", "GL7d18", "GL7d19"], "chain_len": 3, "n": 955128, "ground_truth_nnz": 106019348, "Ranger13-relative-error": 1.0306, "Ranger13-time": 3.07, ...}
```

Default predictors: **Ranger13**, **MNC**, **LGEXP16**, **DensityMap**, **Rose**, **MetaAC**.

### Bundled inputs

| Path | Role |
|------|------|
| `materials/input/SuiteSparse/combinations.jsonl` | ordered matrix-name chains to evaluate |
| `materials/input/SuiteSparse/matrices_catalog.json` | matrix metadata and `.mat` paths |
| `materials/input/SuiteSparse/matrix/*.mat` | SuiteSparse input matrices |

Example `combinations.jsonl` line:

```json
{"combination": ["GL7d17", "GL7d18", "GL7d19"]}
```

Matrix files under `materials/input/SuiteSparse/matrix/` are **not** bundled in the repository; download them before running the experiment.

### Download matrices

Use `materials/input/SuiteSparse/download_matrices.py` to fetch every matrix referenced in `matrices_catalog.json` from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu). Files are saved to the paths given by each entry's `mat_path` field (default: `materials/input/SuiteSparse/matrix/{name}.mat`). Existing files are skipped.

```bash
cd VLDB2026

pip install requests tqdm   # tqdm is optional (progress bar)

python3 materials/input/SuiteSparse/download_matrices.py
```

Optional flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--catalog` | `materials/input/SuiteSparse/matrices_catalog.json` | catalog JSON to read |
| `--repo-root` | repository root (`VLDB2026/`) | base path for relative `mat_path` values |
| `--workers` | `4` | parallel download threads |
| `--max-retries` | `5` | retries per matrix with exponential backoff |

### Dependencies

| Component | Purpose |
|-----------|---------|
| **Python 3** | launch script |
| **NumPy**, **SciPy**, **tqdm** | GT chain multiply, sparse linear algebra |
| **requests** | download SuiteSparse `.mat` files (`download_matrices.py`) |
| **hdf5storage** (or SciPy MAT v7.3 support) | load SuiteSparse `.mat` files |
| **pybind11** + **g++** (C++11/17, **OpenMP**) | compile predictor extensions if `.so` files are missing |

Extensions (under `baseline/` and `script/sparsity_estimate/src/`):

| Module | Build command (from repo root) |
|--------|--------------------------------|
| `mnc`, `DensityMap`, `LG_EXP1`, `rse` (Rose) | `python3 baseline/setup_<name>.py build_ext --inplace` |
| `Ranger13` | `python3 script/sparsity_estimate/src/setup_Ranger13.py build_ext --inplace` |

`baseline/Meta.py` provides pure-NumPy **MetaAC** / **MetaWC** (no compile step).

Install MAT loader dependency if needed:

```bash
pip install hdf5storage
```

```bash
cd VLDB2026

python3 script/sparsity_estimate/run_suite_chain_predictors.py \
  --out experiment/exp_suite_sparse/predictor_errors.jsonl
```

Defaults:

| Flag | Default |
|------|---------|
| `--catalog` | `materials/input/SuiteSparse/matrices_catalog.json` |
| `--combos` | `materials/input/SuiteSparse/combinations.jsonl` |
| `--out` | `experiment/exp_suite_sparse/predictor_errors.jsonl` |
| `--range_r` | `16` (Ranger13 embedding dimension) |
| `--density_b` | `256` (DensityMap block size) |
| `--method-timeout-sec` | `300` |

### Run a subset of predictors

```bash
python3 script/sparsity_estimate/run_suite_chain_predictors.py \
  --methods Rose,Ranger13 \
  --out experiment/exp_suite_sparse/predictor_errors_rose_r13.jsonl
```

Other useful flags:

| Flag | Meaning |
|------|---------|
| `--max_jobs N` | evaluate only the first `N` combinations |
| `--resume` | append to `--out` and skip combinations already written |
| `--methods CSV` | run only listed tags (`Ranger13`, `MNC`, `LGEXP16`, `DensityMap`, `Rose`, `MetaAC`, `MetaWC`) |

---

## Others

* The five datasets used in the paper are available from the following sources:

        DBLP: http://dblp.uni-trier.de/xml/
        DBpedia: https://wiki.dbpedia.org/Datasets
        FourSquare: https://sites.google.com/site/yangdingqi/home/foursquare-dataset
        FreeBase: http://freebase-easy.cs.uni-freiburg.de/dump/
        IMDB: https://www.imdb.com/interfaces/
* The sparse matrices used in the SuiteSparse experiment can be downloaded automatically with `materials/input/SuiteSparse/download_matrices.py`, or manually from https://sparse.tamu.edu.
