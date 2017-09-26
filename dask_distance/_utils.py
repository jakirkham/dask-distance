import dask
import dask.array

from . import _compat


def _bool_cmp_mtx_cnt(u, v):
    u = _compat._asarray(u)
    v = _compat._asarray(v)

    u_1 = u.astype(bool)
    v_1 = v.astype(bool)
    u_0 = ~u_1
    v_0 = ~v_1

    uv_11 = u_1 & v_1
    uv_10 = u_1 & v_0
    uv_01 = u_0 & v_1
    uv_00 = u_0 & v_0

    uv_11_sum = uv_11.sum(axis=0, dtype=float)
    uv_10_sum = uv_10.sum(axis=0, dtype=float)
    uv_01_sum = uv_01.sum(axis=0, dtype=float)
    uv_00_sum = uv_00.sum(axis=0, dtype=float)

    uv_cmp_mtx_cnts = dask.array.stack([
        dask.array.stack([uv_00_sum, uv_01_sum]),
        dask.array.stack([uv_10_sum, uv_11_sum]),
    ])

    return uv_cmp_mtx_cnts
