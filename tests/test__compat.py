#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pytest

import numpy as np

import dask.array as da
import dask.array.utils as dau

import dask_distance._compat


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


@pytest.mark.parametrize("a", [
    2,
    np.array(2),
    np.array([2, 3]),
    np.array([[2], [3]]),
    np.array([[2, 3]]),
    np.array([[2, 3], [4, 5]]),
    np.array([[[2, 3], [4, 5]], [[6, 9], [7, 1]]]),
    [np.array(2), np.array([[2, 3]])],
])
def test_atleast_2d(a):
    if not isinstance(a, list):
        a = [a]

    d = dask_distance._compat._atleast_2d(*a)

    if not isinstance(d, list):
        d = [d]

    for d_i in d:
        assert isinstance(d_i, da.Array)
        assert d_i.ndim >= 2
