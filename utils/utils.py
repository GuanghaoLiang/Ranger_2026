import random

import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def read_metapath_file(filepath, gt=False):
    metapaths = []
    nnz_values = []
    with open(filepath, "r") as f:
        for line in f:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
            try:
                parts = cleaned_line.split()
                metapath = list(map(int, parts[0].split("-")))
                metapaths.append(metapath)
                if gt:
                    nnz_values.append(float(parts[1]))
            except ValueError as exc:
                print(f"ValueError: {exc} in line: {line}")
            except Exception as exc:
                print(f"Error: {exc} in line: {line}")
    if gt:
        return metapaths, nnz_values
    return metapaths


def resume(path):
    existing = set()
    if not path or not __import__("os").path.exists(path):
        return existing
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = eval(line)
            existing.add(item[0])
    return existing
