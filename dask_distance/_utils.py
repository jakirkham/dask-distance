import functools
import itertools

import numpy

import dask
import dask.array

from . import _compat
from . import _pycompat


def _broadcast_uv(u, v):
    U = _compat._atleast_2d(u)
    V = _compat._atleast_2d(v)

    if U.ndim != 2:
        raise ValueError("u must be a 1-D or 2-D array.")
    if V.ndim != 2:
        raise ValueError("v must be a 1-D or 2-D array.")

    U = U[:, None]
    V = V[None, :]

    return U, V


def _unbroadcast_uv(u, v, result):
    u = _compat._asarray(u)
    v = _compat._asarray(v)

    if v.ndim == 1:
        result = result[:, 0]
    if u.ndim == 1:
        result = result[0]

    return result


def _broadcast_uv_wrapper(func):
    @functools.wraps(func)
    def _wrapped_broadcast_uv(u, v):
        U, V = _broadcast_uv(u, v)

        result = func(U, V)

        result = _unbroadcast_uv(u, v, result)

        return result

    return _wrapped_broadcast_uv


def _cdist_apply(U, V, metric):
    U = U[:, 0, :]
    V = V[0, :, :]

    result = numpy.empty((len(U), len(V)), dtype=float)

    for i, j in numpy.ndindex(result.shape):
        result[i, j] = metric(U[i], V[j])

    return result


def _bool_cmp_cnts(U, V):
    U = _compat._asarray(U)
    V = _compat._asarray(V)

    U = U.astype(bool)
    V = V.astype(bool)

    U_01 = [~U, U]
    V_01 = [~V, V]

    UV_cmp_cnts = numpy.empty((2, 2), dtype=object)
    UV_ranges = [_pycompat.irange(e) for e in UV_cmp_cnts.shape]

    for i, j in itertools.product(*UV_ranges):
        UV_cmp_cnts[i, j] = (U_01[i] & V_01[j]).sum(axis=-1, dtype=float)

    for i in _pycompat.irange(UV_cmp_cnts.ndim - 1, -1, -1):
        UV_cmp_cnts2 = UV_cmp_cnts[..., 0]
        for j in itertools.product(*(UV_ranges[:i])):
            UV_cmp_cnts2[j] = dask.array.stack(UV_cmp_cnts[j].tolist(), axis=0)
        UV_cmp_cnts = UV_cmp_cnts2
    UV_cmp_cnts = UV_cmp_cnts[()]

    return UV_cmp_cnts
