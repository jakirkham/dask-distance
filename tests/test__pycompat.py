#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import

import dask_distance._pycompat


def test_irange():
    r = dask_distance._pycompat.irange(5)

    assert not isinstance(r, list)

    assert list(r) == [0, 1, 2, 3, 4]


def test_izip():
    r = dask_distance._pycompat.izip([1, 2, 3], ["a", "b", "c"])

    assert not isinstance(r, list)

    assert list(r) == [(1, 'a'), (2, 'b'), (3, 'c')]
