#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import pytest

import numpy as np

import dask_distance._utils


@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((1, 2, 1,), dtype=bool), np.zeros((2,), dtype=bool)),
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((1, 2, 1,), dtype=bool)),
])
def test__broadcast_uv_err(et, u, v):
    with pytest.raises(et):
        dask_distance._utils._broadcast_uv(u, v)


@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((3,), dtype=bool)),
    (ValueError, np.zeros((1, 2, 1,), dtype=bool), np.zeros((2,), dtype=bool)),
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((1, 2, 1,), dtype=bool)),
])
def test__bool_cmp_cnts_err(et, u, v):
    with pytest.raises(et):
        dask_distance._utils._bool_cmp_cnts(u, v)


def test__bool_cmp_cnts_1d():
    u = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    v = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)

    uv_cmp_mtx = dask_distance._utils._bool_cmp_cnts(u, v)

    uv_cmp_mtx_exp = np.array([[1, 2], [3, 4]], dtype=float)

    assert (np.array(uv_cmp_mtx) == uv_cmp_mtx_exp).all()


def test__bool_cmp_cnts_nd():
    u = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    v = np.array([[0, 1, 1, 0, 0, 0, 1, 1, 1, 1]], dtype=bool)

    uv_cmp_mtx = dask_distance._utils._bool_cmp_cnts(u, v)

    uv_cmp_mtx_exp = np.array([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=float)

    assert (np.array(uv_cmp_mtx) == uv_cmp_mtx_exp).all()
