# -*- coding: utf-8 -*-


import numpy

import dask
import dask.array


def _asarray(a):
    """
    Creates a Dask array based on ``a``.

    Parameters
    ----------
    a : array-like
        Object to convert to a Dask Array.

    Returns
    -------
    a : Dask Array
    """

    if not isinstance(a, dask.array.Array):
        a = numpy.asarray(a)
        a = dask.array.from_array(a, a.shape)

    return a


def _atleast_2d(*arys):
    """
    Provide at least 2-D views of the arrays.

    Parameters
    ----------
    *arys : Dask Array
            Arrays to make at least 2-D

    Returns
    -------
    *res : Dask Array sequence
    """

    result = []
    for a in arys:
        a = _asarray(a)
        if a.ndim == 0:
            a = a[None, None]
        elif a.ndim == 1:
            a = a[None]

        result.append(a)

    if len(arys) == 1:
        result = result[0]

    return result
