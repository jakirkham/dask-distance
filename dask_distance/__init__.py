# -*- coding: utf-8 -*-


from __future__ import division, unicode_literals

import dask
import dask.array

from . import _compat
from . import _utils

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


#######################################
#                                     #
#  Distance matrix functions          #
#                                     #
#######################################


def cdist(XA, XB, metric="euclidean", **kwargs):
    """
    Finds the distance matrix using the metric on each pair of points.

    Args:
        XA:         2-D array of points
        XB:         2-D array of points
        metric:     string or callable
        **kwargs:   provided to the metric (see below)

    Keyword Args:
        p:          p-norm for minkowski only (default: 2)

    Returns:
        array:      distance between each combination of points
    """

    func_mappings = {
        "braycurtis": braycurtis,
        "canberra": canberra,
        "chebyshev": chebyshev,
        "cityblock": cityblock,
        "correlation": correlation,
        "cosine": cosine,
        "dice": dice,
        "euclidean": euclidean,
        "hamming": hamming,
        "jaccard": jaccard,
        "kulsinski": kulsinski,
        "mahalanobis": mahalanobis,
        "minkowski": minkowski,
        "rogerstanimoto": rogerstanimoto,
        "russellrao": russellrao,
        "sokalmichener": sokalmichener,
        "sokalsneath": sokalsneath,
        "seuclidean": seuclidean,
        "sqeuclidean": sqeuclidean,
        "wminkowski": wminkowski,
        "yule": yule,
    }

    result = None
    if callable(metric):
        XA = _compat._asarray(XA)
        XB = _compat._asarray(XB)

        XA_bc, XB_bc = _utils._broadcast_uv(XA, XB)

        XA_bc = XA_bc.rechunk(XA_bc.chunks[:-1] + ((XA_bc.shape[-1],),))
        XB_bc = XB_bc.rechunk(XB_bc.chunks[:-1] + ((XB_bc.shape[-1],),))

        XA_bc = XA_bc.astype(float)
        XB_bc = XB_bc.astype(float)

        result = dask.array.atop(
            _utils._cdist_apply, "ij",
            XA_bc, "ijk",
            XB_bc, "ijk",
            dtype=float,
            concatenate=True,
            metric=metric
        )

        result = result.rechunk(XA.chunks[0:1] + XB.chunks[0:1])
    else:
        try:
            metric = metric.decode("utf-8")
        except AttributeError:
            pass

        metric = func_mappings[metric]

        if metric == mahalanobis:
            if "VI" not in kwargs:
                kwargs["VI"] = (
                    dask.array.linalg.inv(
                        dask.array.cov(dask.array.vstack([XA, XB]).T)
                    ).T
                )
        elif metric == minkowski:
            kwargs.setdefault("p", 2)
        elif metric == seuclidean:
            if "V" not in kwargs:
                kwargs["V"] = (
                    dask.array.var(dask.array.vstack([XA, XB]), axis=0, ddof=1)
                )
        elif metric == wminkowski:
            kwargs.setdefault("p", 2)

        result = metric(XA, XB, **kwargs)

    return result


def pdist(X, metric="euclidean", **kwargs):
    """
    Finds the pairwise condensed distance matrix using the metric.

    Args:
        X:          2-D array of points
        metric:     string or callable
        **kwargs:   provided to the metric (see below)

    Keyword Args:
        p:          p-norm for minkowski only (default: 2)

    Returns:
        array:      condensed distance between each pair

    Note:
        Tries to avoid redundant computations as much as possible.
        However this is limited in its ability to do this based on
        the chunk size of X (particularly along the first dimension).
        Smaller chunks will increase savings though there may be
        other tradeoffs.
    """

    if metric == "mahalanobis":
        if "VI" not in kwargs:
            kwargs["VI"] = dask.array.linalg.inv(dask.array.cov(X.T)).T
    elif metric == "seuclidean":
        if "V" not in kwargs:
            kwargs["V"] = dask.array.var(X, axis=0, ddof=1)

    result = cdist(X, X, metric, **kwargs)

    result = dask.array.triu(result, 1)

    indices = _compat._indices(result.shape, chunks=result.chunks)
    mask = (indices[1] > indices[0])

    result = _compat._ravel(result)[_compat._ravel(mask)]

    return result


#######################################
#                                     #
#  Numeric vector distance functions  #
#                                     #
#######################################


@_utils._broadcast_uv_wrapper
def braycurtis(u, v):
    """
    Finds the Bray-Curtis distance between two 1-D arrays.

    .. math::

       \\frac{ \sum_{i} \lvert u_{i} - v_{i} \\rvert }
             { \sum_{i} \lvert u_{i} + v_{i} \\rvert }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Bray-Curtis distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = abs(u - v).sum(axis=-1) / abs(u + v).sum(axis=-1)

    return result


@_utils._broadcast_uv_wrapper
def canberra(u, v):
    """
    Finds the Canberra distance between two 1-D arrays.

    .. math::

       \sum_{i} \\frac{ \lvert u_{i} - v_{i} \\rvert }
                      { \lvert u_{i} \\rvert + \lvert v_{i} \\rvert }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Canberra distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = (abs(u - v) / (abs(u) + abs(v))).sum(axis=-1)

    return result


@_utils._broadcast_uv_wrapper
def chebyshev(u, v):
    """
    Finds the Chebyshev distance between two 1-D arrays.

    .. math::

       \max_{i} \lvert u_{i} - v_{i} \\rvert

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Chebyshev distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = abs(u - v).max(axis=-1)

    return result


@_utils._broadcast_uv_wrapper
def cityblock(u, v):
    """
    Finds the City Block (Manhattan) distance between two 1-D arrays.

    .. math::

       \sum_{i} \lvert u_{i} - v_{i} \\rvert

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       City Block (Manhattan) distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = abs(u - v).sum(axis=-1)

    return result


@_utils._broadcast_uv_wrapper
def correlation(u, v):
    """
    Finds the correlation distance between two 1-D arrays.

    .. math::

       1 - \\frac{ (u - \\bar{u}) \cdot (v - \\bar{v}) }
                 {
                    \lVert u - \\bar{u} \\rVert_{2}
                    \lVert v - \\bar{v} \\rVert_{2}
                 }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       correlation distance
    """

    u = u.astype(float)
    v = v.astype(float)

    u_mean = u.mean(axis=-1, keepdims=True)
    v_mean = v.mean(axis=-1, keepdims=True)

    result = 1 - (
        ((u - u_mean) * (v - v_mean)).sum(axis=-1) /
        (
            (abs(u - u_mean) ** 2).sum(axis=-1) ** 0.5 *
            (abs(v - v_mean) ** 2).sum(axis=-1) ** 0.5
        )
    )

    return result


@_utils._broadcast_uv_wrapper
def cosine(u, v):
    """
    Finds the Cosine distance between two 1-D arrays.

    .. math::

       1 - \\frac{ u \cdot v }
                 { \lVert u \\rVert_{2} \lVert v \\rVert_{2} }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Cosine distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = 1 - (
        (u * v).sum(axis=-1) /
        (
            (abs(u) ** 2).sum(axis=-1) ** 0.5 *
            (abs(v) ** 2).sum(axis=-1) ** 0.5
        )
    )

    return result


@_utils._broadcast_uv_wrapper
def euclidean(u, v):
    """
    Finds the Euclidean distance between two 1-D arrays.

    .. math::

       \lVert u - v \\rVert_{2}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       Euclidean distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = (abs(u - v) ** 2).sum(axis=-1) ** 0.5

    return result


def mahalanobis(u, v, VI):
    """
    Finds the Mahalanobis distance between two 1-D arrays.

    .. math::

       \sqrt{ (u - v) \cdot V^{-1} \cdot (u - v)^{T} }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays
        V:           Inverse of the covariance matrix

    Returns:
        float:       Mahalanobis distance
    """

    VI = _compat._asarray(VI)
    if VI.ndim != 2:
        raise ValueError("VI must have a dimension of 2.")

    U, V = _utils._broadcast_uv(u, v)

    U = U.astype(float)
    V = V.astype(float)
    VI = VI.astype(float)

    U_sub_V = U - V
    result = dask.array.sqrt(
        (dask.array.tensordot(U_sub_V, VI, axes=1) * U_sub_V).sum(axis=-1)
    )

    result = _utils._unbroadcast_uv(u, v, result)

    return result


def minkowski(u, v, p):
    """
    Finds the Minkowski distance between two 1-D arrays.

    .. math::

       \left( \sum_{i} \lvert u_{i} - v_{i} \\rvert^{p} \\right)^{\\frac{1}{p}}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays
        p:           degree of the norm to use

    Returns:
        float:       Minkowski distance
    """

    U, V = _utils._broadcast_uv(u, v)
    p = _compat._asarray(p)

    U = U.astype(float)
    V = V.astype(float)
    p = p.astype(float)

    result = (abs(U - V) ** p).sum(axis=-1) ** (1 / p)

    result = _utils._unbroadcast_uv(u, v, result)

    return result


def seuclidean(u, v, V):
    """
    Finds the standardized Euclidean distance between two 1-D arrays.

    .. math::

       \sqrt{\sum_{i} \left(
           \\frac{\left( u_{i} - v_{i} \\right)^{2}}{V_{i}}
       \\right)}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays
        V:           1-D array of variances

    Returns:
        float:       standardized Euclidean
    """

    var = V
    del V

    var = _compat._asarray(var)
    if var.ndim != 1:
        raise ValueError("V must have a dimension of 1.")

    U, V = _utils._broadcast_uv(u, v)
    Var = var[None, None].repeat(U.shape[0], axis=0).repeat(U.shape[1], axis=1)

    U = U.astype(float)
    V = V.astype(float)
    Var = Var.astype(float)

    result = dask.array.sqrt(((abs(U - V) ** 2) / Var).sum(axis=-1))

    result = _utils._unbroadcast_uv(u, v, result)

    return result


@_utils._broadcast_uv_wrapper
def sqeuclidean(u, v):
    """
    Finds the squared Euclidean distance between two 1-D arrays.

    .. math::

       \lVert u - v \\rVert_{2}^{2}

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays

    Returns:
        float:       squared Euclidean distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = (abs(u - v) ** 2).sum(axis=-1)

    return result


def wminkowski(u, v, p, w):
    """
    Finds the weighted Minkowski distance between two 1-D arrays.

    .. math::

       \left(
               \sum_{i} \lvert w_{i} \cdot (u_{i} - v_{i}) \\rvert^{p}
        \\right)^{
            \\frac{1}{p}
        }

    Args:
        u:           1-D array or collection of 1-D arrays
        v:           1-D array or collection of 1-D arrays
        p:           degree of the norm to use
        w:           1-D array of weights

    Returns:
        float:       Minkowski distance
    """

    p = _compat._asarray(p)
    w = _compat._asarray(w)

    if w.ndim != 1:
        raise ValueError("w must have a dimension of 1.")

    U, V = _utils._broadcast_uv(u, v)
    W = w[None, None].repeat(U.shape[0], axis=0).repeat(U.shape[1], axis=1)

    U = U.astype(float)
    V = V.astype(float)
    p = p.astype(float)
    W = W.astype(float)

    result = (abs(W * (U - V)) ** p).sum(axis=-1) ** (1 / p)

    result = _utils._unbroadcast_uv(u, v, result)

    return result


#####################################################
#                                                   #
#  Boolean vector distance/dissimilarity functions  #
#                                                   #
#####################################################


@_utils._broadcast_uv_wrapper
def dice(u, v):
    """
    Finds the Dice dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ 2 \cdot c_{TT} + c_{TF} + c_{FT} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Dice dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    uv_mismtch = uv_cmp[1, 0] + uv_cmp[0, 1]

    result = (
        (uv_mismtch) /
        (2 * uv_cmp[1, 1] + uv_mismtch)
    )

    return result


@_utils._broadcast_uv_wrapper
def hamming(u, v):
    """
    Finds the Hamming distance between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ c_{TT} + c_{TF} + c_{FT} + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Hamming distance
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result = (
        (uv_cmp[1, 0] + uv_cmp[0, 1]) / (uv_cmp.sum(axis=(0, 1)))
    )

    return result


@_utils._broadcast_uv_wrapper
def jaccard(u, v):
    """
    Finds the Jaccard-Needham dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ c_{TT} + c_{TF} + c_{FT} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Jaccard-Needham dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    uv_mismtch = uv_cmp[1, 0] + uv_cmp[0, 1]

    result = (
        (uv_mismtch) /
        (uv_cmp[1, 1] + uv_mismtch)
    )

    return result


@_utils._broadcast_uv_wrapper
def kulsinski(u, v):
    """
    Finds the Kulsinski dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ 2 \cdot \left(c_{TF} + c_{FT}\\right) + c_{FF} }
             { c_{TT} + 2 \cdot \left(c_{TF} + c_{FT}\\right) + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Kulsinski dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result_numer = 2 * (uv_cmp[1, 0] + uv_cmp[0, 1]) + uv_cmp[0, 0]

    result = (
        (result_numer) /
        (uv_cmp[1, 1] + result_numer)
    )

    return result


@_utils._broadcast_uv_wrapper
def rogerstanimoto(u, v):
    """
    Finds the Rogers-Tanimoto dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ 2 \cdot \left(c_{TF} + c_{FT}\\right) }
             { c_{TT} + 2 \cdot \left(c_{TF} + c_{FT}\\right) + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Rogers-Tanimoto dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result_numer = 2 * (uv_cmp[1, 0] + uv_cmp[0, 1])

    result = (
        (result_numer) /
        (uv_cmp[1, 1] + result_numer + uv_cmp[0, 0])
    )

    return result


@_utils._broadcast_uv_wrapper
def russellrao(u, v):
    """
    Finds the Russell-Rao dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} + c_{FF} }
             { c_{TT} + c_{TF} + c_{FT} + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Russell-Rao dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result_numer = uv_cmp[1, 0] + uv_cmp[0, 1] + uv_cmp[0, 0]

    result = (
        (result_numer) /
        (uv_cmp[1, 1] + result_numer)
    )

    return result


@_utils._broadcast_uv_wrapper
def sokalmichener(u, v):
    """
    Finds the Sokal-Michener dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ 2 \cdot \left(c_{TF} + c_{FT}\\right) }
             { c_{TT} + 2 \cdot \left(c_{TF} + c_{FT}\\right) + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Sokal-Michener dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result_numer = 2 * (uv_cmp[1, 0] + uv_cmp[0, 1])

    result = (
        (result_numer) /
        (uv_cmp[1, 1] + result_numer + uv_cmp[0, 0])
    )

    return result


@_utils._broadcast_uv_wrapper
def sokalsneath(u, v):
    """
    Finds the Sokal-Sneath dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ 2 \cdot \left(c_{TF} + c_{FT}\\right) }
             { c_{TT} + 2 \cdot \left(c_{TF} + c_{FT}\\right) }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Sokal-Sneath dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    result_numer = 2 * (uv_cmp[1, 0] + uv_cmp[0, 1])

    result = (
        (result_numer) /
        (uv_cmp[1, 1] + result_numer)
    )

    return result


@_utils._broadcast_uv_wrapper
def yule(u, v):
    """
    Finds the Yule dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ 2 \cdot c_{TF} \cdot c_{FT} }
             { c_{TT} \cdot c_{FF} + c_{TF} \cdot c_{FT} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array or collection of 1-D bool arrays
        v:           1-D bool array or collection of 1-D bool arrays

    Returns:
        float:       Yule dissimilarity
    """

    uv_cmp = _utils._bool_cmp_cnts(u, v)

    uv_prod_mismtch = uv_cmp[1, 0] * uv_cmp[0, 1]

    result = (
        (2 * uv_prod_mismtch) /
        (uv_cmp[1, 1] * uv_cmp[0, 0] + uv_prod_mismtch)
    )

    return result
