"""Load SuiteSparse .mat files to SciPy COO (extracted from runexp_sp.py; no torch/baseline deps)."""

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix


def load_mat_sparse_to_coo(mat_path):
    try:
        from scipy.io import loadmat

        d = loadmat(mat_path, squeeze_me=True)
        try:
            return _extract_A_to_coo_dict(d, mat_path, source="scipy")
        except RuntimeError:
            pass
    except NotImplementedError:
        pass

    try:
        import hdf5storage

        d = hdf5storage.loadmat(mat_path, squeeze_me=True)
        return _extract_A_to_coo_dict(d, mat_path, source="hdf5storage")
    except ImportError:
        raise RuntimeError(
            "请安装 hdf5storage: pip install hdf5storage (或 conda install -c conda-forge hdf5storage)"
        ) from None


def _extract_A_to_coo_dict(d, mat_path=None, source="scipy"):
    obj = None
    if "A" in d:
        obj = d["A"]
    elif "Problem" in d:
        P = d["Problem"]
        if isinstance(P, dict):
            obj = P.get("A", None)
        else:
            try:
                names = getattr(P.dtype, "names", None)
                if names and "A" in names:
                    Af = P["A"]
                    try:
                        obj = Af.item()
                    except Exception:
                        obj = Af
                else:
                    obj = getattr(P, "A", None)
            except Exception:
                obj = getattr(P, "A", None)
    if obj is None:
        meta_keys = {"__header__", "__version__", "__globals__"}
        user_keys = [k for k in d.keys() if k not in meta_keys]
        if len(user_keys) == 1:
            obj = d[user_keys[0]]
        else:
            if source == "hdf5storage":
                if not mat_path:
                    raise RuntimeError("未能解析稀疏矩阵，且未提供文件路径用于 HDF5 手动加载")
                return _load_v73_sparse_manual(mat_path)
            raise RuntimeError("未能解析稀疏矩阵变量")
    if hasattr(obj, "tocoo"):
        return obj.tocoo()
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if {"ir", "jc", "data", "shape"} <= keys:
            ir = np.asarray(obj["ir"], dtype=np.int64)
            jc = np.asarray(obj["jc"], dtype=np.int64)
            data = np.asarray(obj["data"])
            shape = tuple(obj["shape"])
            return csc_matrix((data, ir, jc), shape=shape).tocoo()
        if {"ir", "jc", "data"} <= keys:
            ir = np.asarray(obj["ir"], dtype=np.int64)
            jc = np.asarray(obj["jc"], dtype=np.int64)
            data = np.asarray(obj["data"])
            rows = int(ir.max()) + 1 if ir.size else obj.get("rows", 0)
            cols = int(jc.size - 1)
            return csc_matrix((data, ir, jc), shape=(rows, cols)).tocoo()

    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        names = set(obj.dtype.names)
        if {"ir", "jc", "data", "shape"} <= names:
            ir = np.asarray(obj["ir"]).astype(np.int64).ravel()
            jc = np.asarray(obj["jc"]).astype(np.int64).ravel()
            data = np.asarray(obj["data"]).ravel()
            shape = tuple(np.asarray(obj["shape"]).ravel())
            return csc_matrix((data, ir, jc), shape=shape).tocoo()
        if {"ir", "jc", "data"} <= names:
            ir = np.asarray(obj["ir"]).astype(np.int64).ravel()
            jc = np.asarray(obj["jc"]).astype(np.int64).ravel()
            data = np.asarray(obj["data"]).ravel()
            rows = int(ir.max()) + 1 if ir.size else 0
            cols = int(jc.size - 1)
            return csc_matrix((data, ir, jc), shape=(rows, cols)).tocoo()
        try:
            inner = obj.item()
            if isinstance(inner, dict):
                return _extract_A_to_coo_dict({"A": inner}, mat_path, source)
            if hasattr(inner, "tocoo"):
                return inner.tocoo()
        except Exception:
            pass

    try:
        dense = np.asarray(obj)
        if dense.ndim >= 2:
            return coo_matrix(dense)
    except Exception:
        pass
    raise RuntimeError("未能解析稀疏矩阵 'A'")


def _load_v73_sparse_manual(mat_path):
    import h5py

    with h5py.File(mat_path, "r") as f:
        g = None
        if "A" in f:
            g = f["A"]
        elif "Problem" in f and "A" in f["Problem"]:
            g = f["Problem"]["A"]
        if g is None:
            raise RuntimeError("HDF5 中未找到 A 组")
        ir = np.array(g["ir"])
        jc = np.array(g["jc"])
        data = np.array(g["data"])
        rows = (
            int(ir.max()) + 1
            if ir.size
            else int(g["dims"][0])
            if "dims" in g
            else 0
        )
        cols = int(jc.size - 1)
        return csc_matrix((data, ir, jc), shape=(rows, cols)).tocoo()
