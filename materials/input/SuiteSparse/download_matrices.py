#!/usr/bin/env python3
"""Download SuiteSparse .mat files listed in matrices_catalog.json."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SS_ROOT_URL = "https://sparse.tamu.edu"
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
DEFAULT_CATALOG = _SCRIPT_DIR / "matrices_catalog.json"
DEFAULT_MATRIX_DIR = _SCRIPT_DIR / "matrix"


def _mat_url(ss_group: str, ss_name: str) -> str:
    """Same layout as ssgetpy.Matrix.url('MAT'): .../mat/{group}/{name}.mat"""
    return f"{SS_ROOT_URL}/mat/{ss_group}/{ss_name}.mat"


def _resolve_dest(repo_root: Path, entry: dict) -> Path:
    mat_path = entry.get("mat_path")
    if mat_path:
        p = Path(mat_path)
        if p.is_absolute():
            return p
        return repo_root / p
    name = entry["name"]
    return DEFAULT_MATRIX_DIR / f"{name}.mat"


def _stream_download(url: str, dest: Path, max_retries: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=(30, None)) as resp:
                resp.raise_for_status()
                with partial.open("wb") as outfile:
                    for blk in resp.iter_content(chunk_size=8192 * 16):
                        if blk:
                            outfile.write(blk)
            partial.replace(dest)
            return
        except BaseException as exc:
            last_err = exc
            for pth in (partial, dest):
                if pth.is_file():
                    try:
                        pth.unlink()
                    except OSError:
                        pass
            time.sleep(min(60.0, 2**attempt))
    raise RuntimeError(f"download failed after {max_retries} tries: {url} ({last_err})")


def _download_one(name: str, url: str, dest: Path, max_retries: int) -> tuple[str, str | None]:
    if dest.is_file():
        return name, None
    try:
        _stream_download(url, dest, max_retries=max_retries)
        return name, None
    except BaseException as exc:
        return name, str(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download matrices from SuiteSparse for entries in matrices_catalog.json"
    )
    p.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Catalog JSON path (default: {DEFAULT_CATALOG})",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help=f"VLDB2026 repo root for relative mat_path (default: {_REPO_ROOT})",
    )
    p.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    p.add_argument("--max-retries", type=int, default=5, help="Retries per matrix with backoff")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.catalog.open(encoding="utf-8") as f:
        catalog = json.load(f)

    tasks: list[tuple[str, str, Path]] = []
    for entry in catalog:
        name = entry["name"]
        group = entry.get("ss_group")
        ss_name = entry.get("ss_name", name)
        if not group:
            raise KeyError(f"catalog entry {name!r} missing ss_group")
        tasks.append((name, _mat_url(group, ss_name), _resolve_dest(args.repo_root, entry)))

    errors: list[tuple[str, Path, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_map = {
            ex.submit(_download_one, name, url, dest, args.max_retries): (name, dest)
            for name, url, dest in tasks
        }
        itr = as_completed(fut_map)
        if tqdm is not None:
            itr = tqdm(itr, total=len(fut_map), desc="downloads", unit="mat")
        for fut in itr:
            name, dest = fut_map[fut]
            _, err = fut.result()
            if err:
                errors.append((name, dest, err))

    if errors:
        print("Download errors:")
        for name, dest, err in errors:
            print(f"  {name} ({dest}): {err}")
        raise SystemExit(1)
    print(f"All {len(tasks)} catalog matrices downloaded (or already present).")


if __name__ == "__main__":
    main()
