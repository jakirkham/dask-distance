# -*- coding: utf-8 -*-


from __future__ import division

from . import _utils

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"


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
        float:       Cosine distance
    """

    u = u.astype(float)
    v = v.astype(float)

    result = (abs(u - v) ** 2).sum(axis=-1) ** 0.5

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
