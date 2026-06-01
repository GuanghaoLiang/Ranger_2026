#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **_):
        return x


import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

# --- paths (parent process: light imports only; heavy baselines load in subprocess) ---
_EXP_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _EXP_DIR.parents[1]
_SUITE_INPUT = _PKG_ROOT / "materials" / "input" / "SuiteSparse"
_EXP_SRC = _EXP_DIR / "src"
for _p in (_EXP_DIR, _EXP_SRC, _PKG_ROOT):
    sp = str(_p)
    if sp not in sys.path and Path(_p).is_dir():
        sys.path.insert(0, sp)

from mat_loader import load_mat_sparse_to_coo

DEFAULT_METHOD_TIMEOUT_SEC = 300.0

# (json label / column prefix, subprocess tag name)
ALL_PREDICTOR_SPECS: list[tuple[str, str]] = [
    ("Ranger13", "Ranger13"),
    ("MNC", "MNC"),
    ("LGEXP16", "LGEXP16"),
    ("DensityMap", "DensityMap"),
    ("Rose", "Rose"),
    ("MetaAC", "MetaAC"),
    ("MetaWC", "MetaWC"),
]
_KNOWN_METHOD_TAGS = frozenset(tag for _, tag in ALL_PREDICTOR_SPECS)


def _parse_method_csv(s: str) -> frozenset[str]:
    tags = frozenset(x.strip() for x in s.split(",") if x.strip())
    bad = tags - _KNOWN_METHOD_TAGS
    if bad:
        raise SystemExit(f"--methods unknown tags: {sorted(bad)} (known: {sorted(_KNOWN_METHOD_TAGS)})")
    if not tags:
        raise SystemExit("--methods resulted in empty tag set")
    return tags


def _packed_unique_rows_cols(rows: np.ndarray, cols: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = rows.astype(np.int64, copy=False)
    c = cols.astype(np.int64, copy=False)
    if r.size == 0:
        return r, c
    packed = np.unique(r * np.int64(n) + c)
    rr = packed // np.int64(n)
    cc = packed % np.int64(n)
    return rr, cc


def slice_upper_triangle(rows: np.ndarray, cols: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = rows.astype(np.int64, copy=False)
    c = cols.astype(np.int64, copy=False)
    m = (r < n) & (c < n) & (r <= c)
    return _packed_unique_rows_cols(r[m], c[m], n)


def build_index_list(
    caches: dict[str, tuple[np.ndarray, np.ndarray, int, int]],
    names: tuple[str, ...],
    n: int,
):
    slices = []
    for name in names:
        rs, cs, nr, nc = caches[name]
        if n > min(nr, nc):
            raise ValueError(name)
        r, c = slice_upper_triangle(rs, cs, n)
        slices.append([r.astype(np.int32, copy=False), c.astype(np.int32, copy=False)])
    return slices


def csr_upper(rs: np.ndarray, cs: np.ndarray, n: int):
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


def ground_truth_nnz_scipy(n: int, index_list: list) -> int:
    mats = []
    for row, col in index_list:
        mats.append(
            csr_upper(
                np.asarray(row, dtype=np.int64),
                np.asarray(col, dtype=np.int64),
                n,
            )
        )
    if not mats:
        return 0
    acc = mats[0]
    for j in range(1, len(mats)):
        acc = bool_mul_support(acc, mats[j])
    return acc.nnz


def ground_truth_from_mat(mat_path: str) -> int:
    coo = load_mat_sparse_to_coo(mat_path)
    return int(coo.nnz)


def relative_error(est: float, gt: float) -> float | None:
    if est <= 0 or gt <= 0:
        return None
    return float(max(est, gt) / min(est, gt))


def four(x: float | None) -> float | None:
    if x is None:
        return None
    return round(float(x), 4)


def pack_idx_for_subprocess(idx: list) -> tuple:
    """Pickle-friendly immutable snapshot for spawn children."""
    return tuple(
        (np.asarray(rows, dtype=np.int64).copy(), np.asarray(cols, dtype=np.int64).copy())
        for rows, cols in idx
    )


def _spawn_sys_path_bundle() -> list[str]:
    return [
        str(_EXP_DIR),
        str(_EXP_SRC),
        str(_PKG_ROOT),
    ]


def _isolate_worker(conn_end, bundle: dict) -> None:
    """Heavy imports + one estimator; sends ('ok', val) | ('err', msg)."""
    try:
        for sp in bundle["paths"]:
            if sp not in sys.path:
                sys.path.insert(0, sp)

        import numpy as np

        tag = bundle["tag"]
        n = int(bundle["n"])
        range_r = int(bundle["range_r"])
        density_b = int(bundle["density_b"])
        idx_pack = bundle["idx_pack"]
        idx: list[list] = [[pair[0].astype(np.int32), pair[1].astype(np.int32)] for pair in idx_pack]

        # Meta*: pure numpy + baseline.Meta — skip torch/Ranger imports (spawn startup cost.)
        if tag == "MetaAC":
            from baseline.Meta import MetaAC  # noqa: WPS433

            est = float(MetaAC(n, idx)[-1] * n * n)
            conn_end.send(("ok", est))
            return

        if tag == "MetaWC":
            from baseline.Meta import MetaWC  # noqa: WPS433

            est = float(MetaWC(n, idx)[-1] * n * n)
            conn_end.send(("ok", est))
            return

        if tag == "Rose":
            import baseline.rse as rse  # noqa: WPS433

            idx_c = [[np.array(p[0], copy=True), np.array(p[1], copy=True)] for p in idx]
            dens = float(rse.rose_chain(n, idx_c, full=False))
            est = dens * (n * n)
            conn_end.send(("ok", est))
            return

        if tag == "Ranger13":
            from Ranger13 import CRange as crange13_cpp  # noqa: WPS433

            est, _ = crange13_cpp(n, idx, range_r, False, True)
            conn_end.send(("ok", float(est)))
            return

        import baseline.LG_EXP1 as LG_EXP1
        import baseline.mnc as mnc
        import baseline.DensityMap as DensityMap

        if tag == "MNC":
            arr = mnc.mnc_chain(n, idx)
            est = float(arr[-1])
        elif tag == "LGEXP16":
            idx_c = [[np.array(p[0], copy=True), np.array(p[1], copy=True)] for p in idx]
            res = LG_EXP1.layer_graph_chain(n, idx_c, 16)
            est = float(res[-1] * n * n)
        elif tag == "DensityMap":
            res = DensityMap.dm_chain(n, idx, density_b)
            est = float(res[-1])
        else:
            conn_end.send(("err", f"unknown tag {tag!r}"))
            return

        conn_end.send(("ok", est))
    except BaseException as e:
        try:
            conn_end.send(("err", repr(e)))
        except BaseException:
            pass
    finally:
        try:
            conn_end.close()
        except Exception:
            pass


def run_predictor_spawn(
    tag: str,
    n: int,
    idx: list,
    *,
    timeout_sec: float,
    range_r: int,
    density_b: int,
) -> tuple[float | None, float | None]:
    """Return (estimate, wall_sec), or (None, None) if timeout; (None, wall) on child error/crash."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    bundle = {
        "paths": _spawn_sys_path_bundle(),
        "tag": tag,
        "n": n,
        "idx_pack": pack_idx_for_subprocess(idx),
        "range_r": range_r,
        "density_b": density_b,
    }
    proc = ctx.Process(target=_isolate_worker, args=(child_conn, bundle))
    proc.daemon = True
    proc.start()

    t0 = time.perf_counter()
    proc.join(timeout=float(timeout_sec))
    wall_s = time.perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        return None, None

    payload = None
    try:
        if parent_conn.poll():
            payload = parent_conn.recv()
    except (BrokenPipeError, EOFError, OSError):
        payload = None
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass

    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] == "ok":
        return float(payload[1]), four(wall_s)

    return None, four(wall_s)


def _resolve_mat_path(mat_path: str) -> str:
    p = Path(mat_path)
    if p.is_file():
        return str(p)
    rooted = _PKG_ROOT / mat_path
    if rooted.is_file():
        return str(rooted)
    return mat_path


def _load_matrix_entry_cached(entry: dict, cache: dict) -> None:
    name = entry["name"]
    if name in cache:
        return
    coo = load_mat_sparse_to_coo(_resolve_mat_path(entry["mat_path"]))
    cache[name] = (
        np.asarray(coo.row, dtype=np.int32),
        np.asarray(coo.col, dtype=np.int32),
        int(coo.shape[0]),
        int(coo.shape[1]),
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="SuiteSparse chains (k>=2) × predictor errors (subprocess + per-method timeout)"
    )
    p.add_argument(
        "--catalog",
        type=str,
        default=str(_SUITE_INPUT / "matrices_catalog.json"),
    )
    p.add_argument(
        "--combos",
        type=str,
        default=str(_SUITE_INPUT / "combinations.jsonl"),
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(_PKG_ROOT / "experiment" / "exp_suite_sparse" / "predictor_errors.jsonl"),
    )
    p.add_argument(
        "--range_r",
        type=int,
        default=16,
        help="Embedding r for Ranger13 backend (default 16)",
    )
    p.add_argument("--density_b", type=int, default=256)
    p.add_argument(
        "--method-timeout-sec",
        type=float,
        default=DEFAULT_METHOD_TIMEOUT_SEC,
        metavar="SEC",
        dest="method_timeout_sec",
        help=f"Kill subprocess if still running after SEC (default {DEFAULT_METHOD_TIMEOUT_SEC:g}). "
        "On timeout both *-relative-error and *-time become null.",
    )
    p.add_argument("--max_jobs", type=int, default=None)
    p.add_argument("--preload_workers", type=int, default=8)
    p.add_argument(
        "--require_cubool_mat",
        action="store_true",
        help="Skip combinations whose output_mat .mat does not exist (uses cubool file as GT)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to --out and skip combinations already present (matched by full ordered name list)",
    )
    p.add_argument(
        "--merge-update-from",
        type=str,
        default="",
        metavar="PATH",
        help="Read PATH JSONL line-by-line; recompute only tags from --methods (default MetaAC,MetaWC) "
        "and write full merged rows to --out. Ignores --combos.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="",
        metavar="CSV",
        help="Comma-separated method tags to run alone (standalone) or patch (with --merge-update-from). "
        "Standalone default: all. Merge default: MetaAC,MetaWC.",
    )
    return p.parse_args()


def _json_combo_key(o: dict) -> tuple[str, ...] | None:
    if o.get("skipped"):
        return None
    try:
        return tuple(str(x) for x in o["combination"])
    except (KeyError, TypeError):
        return None


def _load_done_combos(out_path: Path) -> set[tuple[str, ...]]:
    if not out_path.is_file():
        return set()
    done: set[tuple[str, ...]] = set()
    try:
        with out_path.open(encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("skipped"):
                    continue
                t = o["combination"]
                done.add(tuple(str(x) for x in t))
    except (json.JSONDecodeError, KeyError):
        pass
    return done


def evaluate_one_combo(
    combo: tuple[str, ...],
    output_mat_path: str | None,
    caches: dict[str, tuple],
    *,
    range_r: int,
    density_b: int,
    require_cubool_mat: bool,
    method_timeout_sec: float,
    method_tags: frozenset[str] | None = None,
) -> dict:
    if len(combo) < 2:
        raise ValueError("combo must have length >= 2")
    n = min(min(caches[name][2], caches[name][3]) for name in combo)
    idx = build_index_list(caches, combo, n)

    if output_mat_path and os.path.isfile(output_mat_path):
        gt = ground_truth_from_mat(output_mat_path)
    else:
        if require_cubool_mat:
            return {
                "skipped": True,
                "reason": "missing_cubool_mat",
                "combination": list(combo),
                "chain_len": len(combo),
            }
        gt = ground_truth_nnz_scipy(n, idx)

    rec: dict = {
        "combination": list(combo),
        "chain_len": len(combo),
        "n": int(n),
        "ground_truth_nnz": int(gt),
    }
    if output_mat_path:
        rec["output_mat"] = output_mat_path

    specs = ALL_PREDICTOR_SPECS
    if method_tags is not None:
        specs = [(lb, tg) for lb, tg in specs if tg in method_tags]

    gt_f = float(gt)
    for label, tag in specs:
        k_e, k_t = f"{label}-relative-error", f"{label}-time"
        est, wt = run_predictor_spawn(
            tag,
            n,
            idx,
            timeout_sec=method_timeout_sec,
            range_r=range_r,
            density_b=density_b,
        )
        if est is None and wt is None:
            rec[k_e] = None
            rec[k_t] = None
        elif est is None:
            rec[k_e] = None
            rec[k_t] = wt
        else:
            rec[k_e] = four(relative_error(est, gt_f))
            rec[k_t] = wt

    return rec


def main():
    args = parse_args()

    merge_src = str(args.merge_update_from or "").strip()
    method_csv = str(args.methods or "").strip()

    if merge_src and method_csv:
        method_subset = _parse_method_csv(method_csv)
    elif merge_src:
        method_subset = frozenset({"MetaAC", "MetaWC"})
    elif method_csv:
        method_subset = _parse_method_csv(method_csv)
    else:
        method_subset = None

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)

    caches: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.preload_workers)) as ex:
        futs = [ex.submit(_load_matrix_entry_cached, e, caches) for e in catalog]
        for fut in as_completed(futs):
            fut.result()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mtd_to = float(args.method_timeout_sec)

    if merge_src:
        in_path = Path(merge_src)
        if not in_path.is_file():
            raise SystemExit(f"--merge-update-from missing or not file: {in_path}")

        lines_in: list[str] = []
        with in_path.open(encoding="utf-8") as rf:
            for line in rf:
                if line.strip():
                    lines_in.append(line.rstrip("\n"))

        if args.max_jobs is not None:
            lines_in = lines_in[: max(0, args.max_jobs)]

        if args.resume:
            done = _load_done_combos(out_path)
            filtered: list[str] = []
            for li in lines_in:
                key = _json_combo_key(json.loads(li))
                if key is not None and key not in done:
                    filtered.append(li)
            if (skipped_n := len(lines_in) - len(filtered)) > 0:
                print(f"Resume: skipping {skipped_n} combinations already in {out_path}")
            lines_in = filtered

        mode = "a" if args.resume and out_path.is_file() else "w"
        specs_run = [(lb, tg) for lb, tg in ALL_PREDICTOR_SPECS if tg in method_subset]
        processed = 0

        with out_path.open(mode, encoding="utf-8") as out_f:
            for line in tqdm(lines_in, desc="merge-update", unit="combo"):
                try:
                    base = json.loads(line)
                    if base.get("skipped"):
                        continue
                    tri = tuple(str(x) for x in base["combination"])
                    n_saved = int(base["n"])
                    idx = build_index_list(caches, tri, n_saved)
                    gt_f = float(base["ground_truth_nnz"])
                    rec = dict(base)
                    for label, tag in specs_run:
                        k_e, k_t = f"{label}-relative-error", f"{label}-time"
                        est, wt = run_predictor_spawn(
                            tag,
                            n_saved,
                            idx,
                            timeout_sec=mtd_to,
                            range_r=args.range_r,
                            density_b=args.density_b,
                        )
                        if est is None and wt is None:
                            rec[k_e] = None
                            rec[k_t] = None
                        elif est is None:
                            rec[k_e] = None
                            rec[k_t] = wt
                        else:
                            rec[k_e] = four(relative_error(est, gt_f))
                            rec[k_t] = wt
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed += 1
                except Exception as e:
                    tri = json.loads(line).get("combination", [])
                    print(json.dumps({"combination": tri, "error": repr(e)}, ensure_ascii=False))
                    continue

        print(
            f"merge-update-from {in_path}: wrote {processed} line(s) to {out_path} "
            f"(resume={args.resume}, methods={sorted(method_subset)})"
        )
        return

    rows: list[tuple[list[str], str | None]] = []
    with open(args.combos, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = o["combination"]
            rows.append((list(t), o.get("output_mat")))

    if args.max_jobs is not None:
        rows = rows[: max(0, args.max_jobs)]

    if args.resume:
        done = _load_done_combos(out_path)
        todo = [(tl, om) for tl, om in rows if tuple(tl) not in done]
        if (skipped_n := len(rows) - len(todo)):
            print(f"Resume: skipping {skipped_n} combinations already in {out_path}")
    else:
        todo = rows

    mode = "a" if args.resume and out_path.is_file() else "w"
    processed = 0
    with out_path.open(mode, encoding="utf-8") as out_f:
        for tri_list, out_m in tqdm(todo, desc="predictors", unit="combo"):
            tri = tuple(str(x) for x in tri_list)
            try:
                rec = evaluate_one_combo(
                    tri,
                    out_m,
                    caches,
                    range_r=args.range_r,
                    density_b=args.density_b,
                    require_cubool_mat=args.require_cubool_mat,
                    method_timeout_sec=mtd_to,
                    method_tags=method_subset,
                )
                if rec.get("skipped"):
                    continue
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                processed += 1
            except Exception as e:
                print(json.dumps({"combination": list(tri), "error": repr(e)}, ensure_ascii=False))
                continue

    print(f"Wrote {processed} new line(s) to {out_path} (resume={args.resume})")


if __name__ == "__main__":
    mp.freeze_support()
    main()