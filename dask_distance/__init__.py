# -*- coding: utf-8 -*-


from __future__ import division


__author__ = """John Kirkham"""
__email__ = "kirkhamj@janelia.hhmi.org"

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


from . import _utils


def dice(u, v):
    """
    Finds the Dice dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ 2 \cdot c_{TT} + c_{TF} + c_{FT} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array
        v:           1-D bool array

    Returns:
        float:       Dice dissimilarity.
    """

    uv_mtx = _utils._bool_cmp_mtx_cnt(u, v)

    result = (
        (uv_mtx[1, 0] + uv_mtx[0, 1]) /
        (2 * uv_mtx[1, 1] + uv_mtx[1, 0] + uv_mtx[0, 1])
    )

    return result


def hamming(u, v):
    """
    Finds the Hamming distance between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ c_{TT} + c_{TF} + c_{FT} + c_{FF} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array
        v:           1-D bool array

    Returns:
        float:       Hamming distance
    """

    uv_mtx = _utils._bool_cmp_mtx_cnt(u, v)

    result = (
        (uv_mtx[1, 0] + uv_mtx[0, 1]) / (uv_mtx.sum())
    )

    return result


def jaccard(u, v):
    """
    Finds the Jaccard-Needham dissimilarity between two 1-D bool arrays.

    .. math::

       \\frac{ c_{TF} + c_{FT} }{ c_{TT} + c_{TF} + c_{FT} }

    where :math:`c_{XY} = \sum_{i} \delta_{u_{i} X} \delta_{v_{i} Y}`

    Args:
        u:           1-D bool array
        v:           1-D bool array

    Returns:
        float:       Jaccard-Needham dissimilarity
    """

    uv_mtx = _utils._bool_cmp_mtx_cnt(u, v)

    result = (
        (uv_mtx[1, 0] + uv_mtx[0, 1]) /
        (uv_mtx[1, 1] + uv_mtx[1, 0] + uv_mtx[0, 1])
    )

    return result
