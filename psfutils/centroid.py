# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Centroid utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['local_centroid']


def _local_centroid1d(data):
    numer = (data[2] - data[0]) / 2.
    denom = data[1] - np.min([data[0], data[2]])
    if denom == 0:
        return 0.    # the center pixel
    else:
        return (numer / denom)


def local_centroid(data, max_position=None):
    """
    Calculate the local centroid of a 2D array using Jay Anderson's
    method.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    max_position : 2-tuple of float, optional
        The (y, x) position of the maximum pixel in the 2D array.  If
        `None`, then it will be calculated.

    Returns
    -------
    xcen, ycen : float
        (x, y) coordinates of the centroid.
    """

    if max_position is None:
        ymax, xmax = np.unravel_index(np.argmax(data), data.shape)
    else:
        ymax, xmax = max_position

    xdata = data[ymax, xmax-1:xmax+2]
    ydata = data[ymax-1:ymax+2, xmax]
    if len(xdata) != 3 or len(ydata) != 3:
        raise ValueError('max_position cannot be on the data edge')

    return (_local_centroid1d(xdata) + xmax,
            _local_centroid1d(ydata) + ymax)
