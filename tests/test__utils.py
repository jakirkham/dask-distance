#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import pytest

import numpy as np

import dask_distance._utils


def test_import_toplevel():
    u = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    v = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)

    uv_cmp_mtx = dask_distance._utils._bool_cmp_mtx_cnt(u, v)

    uv_cmp_mtx_exp = np.array([[1, 2], [3, 4]], dtype=float)

    assert (np.array(uv_cmp_mtx) == uv_cmp_mtx_exp).all()
