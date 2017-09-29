#!/usr/bin/env python
# -*- coding: utf-8 -*-


import distutils.version as ver

import pytest

import numpy as np

import dask
import dask.array as da
import dask.array.utils as dau

import dask_distance._compat


old_dask = ver.LooseVersion(dask.__version__) <= ver.LooseVersion("0.13.0")


@pytest.mark.parametrize("x", [
    list(range(5)),
    np.random.randint(10, size=(15, 16)),
    da.random.randint(10, size=(15, 16), chunks=(5, 5)),
])
def test_asarray(x):
    d = dask_distance._compat._asarray(x)

    assert isinstance(d, da.Array)

    if not isinstance(x, (np.ndarray, da.Array)):
        x = np.asarray(x)

    dau.assert_eq(d, x)


def test_indices_no_chunks():
    with pytest.raises(ValueError):
        dask_distance._compat._indices((1,))


def test_indices_wrong_chunks():
    with pytest.raises(ValueError):
        dask_distance._compat._indices((1,), chunks=tuple())


@pytest.mark.parametrize(
    "dimensions, dtype, chunks",
    [
        (tuple(), int, tuple()),
        (tuple(), float, tuple()),
        ((0,), float, (1,)),
        ((0, 1, 2), float, (1, 1, 2)),
    ]
)
def test_empty_indicies(dimensions, dtype, chunks):
    darr = dask_distance._compat._indices(dimensions, dtype, chunks=chunks)
    nparr = np.indices(dimensions, dtype)

    assert darr.shape == nparr.shape
    assert darr.dtype == nparr.dtype

    try:
        dau.assert_eq(darr, nparr)
    except IndexError:
        if len(dimensions) and old_dask:
            pytest.skip(
                "Dask pre-0.14.0 is unable to compute this empty array."
            )
        else:
            raise


def test_indicies():
    darr = dask_distance._compat._indices((1,), chunks=(1,))
    nparr = np.indices((1,))
    dau.assert_eq(darr, nparr)

    darr = dask_distance._compat._indices((1,), float, chunks=(1,))
    nparr = np.indices((1,), float)
    dau.assert_eq(darr, nparr)

    darr = dask_distance._compat._indices((2, 1), chunks=(2, 1))
    nparr = np.indices((2, 1))
    dau.assert_eq(darr, nparr)

    darr = dask_distance._compat._indices((2, 3), chunks=(1, 2))
    nparr = np.indices((2, 3))
    dau.assert_eq(darr, nparr)


@pytest.mark.parametrize(
    "shape, dtype, chunks",
    [
        ((10, 11, 12), int, (3, 5, 5)),
        ((10, 11, 12), float, (3, 5, 5)),
        ((10, 11, 12), float, (3, 2, 2)),
        ((20, 17, 31), float, (6, 5, 10)),
    ]
)
@pytest.mark.parametrize(
    "seed",
    [
        153,
    ]
)
def test_ravel(shape, dtype, chunks, seed):
    np.random.random(seed)

    a = np.random.randint(0, 10, shape).astype(dtype)
    d = da.from_array(a, chunks=chunks)

    r_a = np.ravel(a)
    r_d = dask_distance._compat._ravel(d)

    dau.assert_eq(r_d, r_a)
