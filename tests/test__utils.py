#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import pytest

import numpy as np

import dask.array as da

import dask_distance._utils


@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((1, 2, 1,), dtype=bool), np.zeros((2,), dtype=bool)),
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((1, 2, 1,), dtype=bool)),
])
def test__broadcast_uv_err(et, u, v):
    with pytest.raises(et):
        dask_distance._utils._broadcast_uv(u, v)


def test__broadcast_uv():
    u = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    v = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

    U, V = dask_distance._utils._broadcast_uv(u, v)

    for each in [U, V]:
        assert isinstance(each, da.core.Array)

    for new, old in [[U, u], [V, v]]:
        assert new.dtype == old.dtype

    assert U.shape == V.shape


def test__unbroadcast_uv():
    u = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    v = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1])

    U, V = dask_distance._utils._broadcast_uv(u, v)

    result = U + V

    result = dask_distance._utils._unbroadcast_uv(u, v, result)

    assert isinstance(result, da.core.Array)

    assert result.shape == u.shape


@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((3,), dtype=bool)),
])
def test__bool_cmp_cnts_err(et, u, v):
    with pytest.raises(et):
        dask_distance._utils._bool_cmp_cnts(u, v)


def test__bool_cmp_cnts_nd():
    u = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    v = np.array([[0, 1, 1, 0, 0, 0, 1, 1, 1, 1]], dtype=bool)

    uv_cmp = dask_distance._utils._bool_cmp_cnts(
        np.repeat(u[:, None], len(v), axis=1),
        np.repeat(v[None], len(u), axis=0)
    )

    uv_cmp_exp = np.array([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=float)

    assert (np.array(uv_cmp) == uv_cmp_exp).all()
