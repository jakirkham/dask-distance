#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import scipy.spatial.distance as spdist

import dask.array as da

import pytest

import dask_distance


@pytest.mark.parametrize(
    "funcname", [
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "rogerstanimoto",
    ]
)
@pytest.mark.parametrize(
    "seed", [
        0,
        137,
        34,
    ]
)
@pytest.mark.parametrize(
    "size, chunks", [
        (10, 5),
    ]
)
def test_1d_bool_dist(funcname, seed, size, chunks):
    np.random.seed(seed)

    a_u = np.random.randint(0, 2, (size,), dtype=bool)
    a_v = np.random.randint(0, 2, (size,), dtype=bool)

    d_u = da.from_array(a_u, chunks=chunks)
    d_v = da.from_array(a_v, chunks=chunks)

    sp_func = getattr(spdist, funcname)
    da_func = getattr(dask_distance, funcname)

    a_r = sp_func(a_u, a_v)
    d_r = da_func(d_u, d_v)

    assert np.array(d_r) == a_r