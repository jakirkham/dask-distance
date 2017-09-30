#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np
import scipy.spatial.distance as spdist

import dask.array as da

import pytest

import dask_distance


@pytest.mark.parametrize(
    "funcname, kw", [
        ("braycurtis", {}),
        ("canberra", {}),
        ("chebyshev", {}),
        ("cityblock", {}),
        ("correlation", {}),
        ("cosine", {}),
        ("euclidean", {}),
        ("mahalanobis", {"VI": 1}),
        ("minkowski", {"p": 3}),
        ("minkowski", {"p": 1.4}),
        ("seuclidean", {"V": 1}),
        ("sqeuclidean", {}),
        ("wminkowski", {"p": 3, "w": 1}),
    ]
)
@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((3,), dtype=bool)),
    (ValueError, np.zeros((1, 2, 1,), dtype=bool), np.zeros((3,), dtype=bool)),
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((1, 3, 1,), dtype=bool)),
])
def test_1d_dist_err(funcname, kw, et, u, v):
    da_func = getattr(dask_distance, funcname)

    with pytest.raises(et):
        da_func(u, v, **kw)


@pytest.mark.parametrize(
    "funcname, kw", [
        ("braycurtis", {}),
        ("canberra", {}),
        ("chebyshev", {}),
        ("cityblock", {}),
        ("correlation", {}),
        ("cosine", {}),
        ("euclidean", {}),
        ("mahalanobis", {}),
        ("minkowski", {"p": 3}),
        ("minkowski", {"p": 1.4}),
        ("seuclidean", {}),
        ("sqeuclidean", {}),
        ("wminkowski", {"p": 4}),
        ("wminkowski", {"p": 1.6}),
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
def test_1d_dist(funcname, kw, seed, size, chunks):
    np.random.seed(seed)

    a_u = 2 * np.random.random((size,)) - 1
    a_v = 2 * np.random.random((size,)) - 1

    d_u = da.from_array(a_u, chunks=chunks)
    d_v = da.from_array(a_v, chunks=chunks)

    sp_func = getattr(spdist, funcname)
    da_func = getattr(dask_distance, funcname)

    if funcname == "mahalanobis":
        kw["VI"] = 2 * np.random.random((size, size)) - 1
    elif funcname == "seuclidean":
        kw["V"] = 2 * np.random.random((size,)) - 1
    elif funcname == "wminkowski":
        kw["w"] = 2 * np.random.random((size,)) - 1

    a_r = sp_func(a_u, a_v, **kw)
    d_r = da_func(d_u, d_v, **kw)

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)


@pytest.mark.parametrize(
    "metric, kw", [
        ("braycurtis", {}),
        ("canberra", {}),
        ("chebyshev", {}),
        ("cityblock", {}),
        ("correlation", {}),
        ("cosine", {}),
        ("euclidean", {}),
        ("mahalanobis", {"VI": None}),
        ("mahalanobis", {}),
        ("minkowski", {}),
        ("minkowski", {"p": 3}),
        ("seuclidean", {"V": None}),
        ("seuclidean", {}),
        ("sqeuclidean", {}),
        ("wminkowski", {}),
        ("wminkowski", {"p": 1.6}),
        (lambda u, v: (abs(u - v) ** 3).sum() ** (1.0 / 3.0), {}),
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
    "u_shape, u_chunks, v_shape, v_chunks", [
        ((7, 6), (2, 2), (3, 6), (2, 2)),
    ]
)
def test_2d_cdist(metric, kw, seed, u_shape, u_chunks, v_shape, v_chunks):
    np.random.seed(seed)

    a_u = 2 * np.random.random(u_shape) - 1
    a_v = 2 * np.random.random(v_shape) - 1

    d_u = da.from_array(a_u, chunks=u_chunks)
    d_v = da.from_array(a_v, chunks=v_chunks)

    if metric == "mahalanobis":
        if "VI" not in kw:
            kw["VI"] = 2 * np.random.random(2 * u_shape[-1:]) - 1
        elif kw["VI"] is None:
            kw.pop("VI")
    elif metric == "seuclidean":
        if "V" not in kw:
            kw["V"] = 2 * np.random.random(u_shape[-1:]) - 1
        elif kw["V"] is None:
            kw.pop("V")
    elif metric == "wminkowski":
        kw["w"] = np.random.random(u_shape[-1:])

    a_r = spdist.cdist(a_u, a_v, metric, **kw)
    d_r = dask_distance.cdist(d_u, d_v, metric, **kw)

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)


@pytest.mark.parametrize(
    "metric, kw", [
        ("braycurtis", {}),
        ("canberra", {}),
        ("chebyshev", {}),
        ("cityblock", {}),
        ("correlation", {}),
        ("cosine", {}),
        ("euclidean", {}),
        ("mahalanobis", {"VI": None}),
        ("mahalanobis", {}),
        ("minkowski", {}),
        ("minkowski", {"p": 3}),
        ("seuclidean", {"V": None}),
        ("seuclidean", {}),
        ("sqeuclidean", {}),
        ("wminkowski", {}),
        ("wminkowski", {"p": 1.6}),
        (lambda u, v: (abs(u - v) ** 3).sum() ** (1.0 / 3.0), {}),
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
    "u_shape, u_chunks", [
        ((7, 6), (2, 3)),
    ]
)
def test_2d_pdist(metric, kw, seed, u_shape, u_chunks):
    np.random.seed(seed)

    a_u = 2 * np.random.random(u_shape) - 1
    d_u = da.from_array(a_u, chunks=u_chunks)

    if metric == "mahalanobis":
        if "VI" not in kw:
            kw["VI"] = 2 * np.random.random(2 * u_shape[-1:]) - 1
        elif kw["VI"] is None:
            kw.pop("VI")
    elif metric == "seuclidean":
        if "V" not in kw:
            kw["V"] = 2 * np.random.random(u_shape[-1:]) - 1
        elif kw["V"] is None:
            kw.pop("V")
    elif metric == "wminkowski":
        kw["w"] = np.random.random(u_shape[-1:])

    a_r = spdist.pdist(a_u, metric, **kw)
    d_r = dask_distance.pdist(d_u, metric, **kw)

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)


@pytest.mark.parametrize(
    "funcname", [
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ]
)
@pytest.mark.parametrize("et, u, v", [
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((3,), dtype=bool)),
    (ValueError, np.zeros((1, 2, 1,), dtype=bool), np.zeros((3,), dtype=bool)),
    (ValueError, np.zeros((2,), dtype=bool), np.zeros((1, 3, 1,), dtype=bool)),
])
def test_1d_bool_dist_err(funcname, et, u, v):
    da_func = getattr(dask_distance, funcname)

    with pytest.raises(et):
        da_func(u, v)


@pytest.mark.parametrize(
    "funcname", [
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
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

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)


@pytest.mark.parametrize(
    "metric", [
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
        lambda u, v: (abs(u - v) ** 3).sum() ** (1.0 / 3.0),
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
    "u_shape, u_chunks, v_shape, v_chunks", [
        ((2, 10), (1, 5), (3, 10), (1, 5)),
    ]
)
def test_2d_bool_cdist(metric, seed, u_shape, u_chunks, v_shape, v_chunks):
    np.random.seed(seed)

    a_u = np.random.randint(0, 2, u_shape, dtype=bool)
    a_v = np.random.randint(0, 2, v_shape, dtype=bool)

    d_u = da.from_array(a_u, chunks=u_chunks)
    d_v = da.from_array(a_v, chunks=v_chunks)

    a_r = spdist.cdist(a_u, a_v, metric)
    d_r = dask_distance.cdist(d_u, d_v, metric)

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)


@pytest.mark.parametrize(
    "metric", [
        "dice",
        "hamming",
        "jaccard",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
        lambda u, v: (abs(u - v) ** 3).sum() ** (1.0 / 3.0),
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
    "u_shape, u_chunks", [
        ((3, 10), (1, 5)),
    ]
)
def test_2d_bool_pdist(metric, seed, u_shape, u_chunks):
    np.random.seed(seed)

    a_u = np.random.randint(0, 2, u_shape, dtype=bool)
    d_u = da.from_array(a_u, chunks=u_chunks)

    a_r = spdist.pdist(a_u, metric)
    d_r = dask_distance.pdist(d_u, metric)

    assert np.allclose(np.array(d_r)[()], a_r, equal_nan=True)
