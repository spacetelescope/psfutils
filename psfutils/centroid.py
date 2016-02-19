# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Centroid utilities.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['local_centroid']


def _jay_local_centroid1d(data):
    data = np.asanyarray(data)
    numer = (data[2] - data[0]) / 2.
    denom = data[1] - np.min([data[0], data[2]])
    if denom == 0:
        return 0.    # the center pixel
    else:
        return (numer / denom)


def _1d_local_centroid1d(data, submin=False):
    data = np.asanyarray(data)
    if submin:
        data = data - data.min()
    return (data[2] - data[0]) / data.sum()


def local_centroid(data, mode='jay', max_position=None):
    """
    Calculate the local centroid of a 2D array using the 1D marginal x
    and y centroids of the central three pixels.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    mode : {'jay', '1d', '1dsub'}, optional
        The centroid algorithm to use:

            * ``'jay'``: Jay Anderson's algorithm (default).
            * ``'1d'``: Standard 1D moment.
            * ``'1dsubmin'``: Standard 1D moment after subtraction of
              the minimum pixel value.

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

    if mode == 'jay':
        xc = _jay_local_centroid1d(xdata)
        yc = _jay_local_centroid1d(ydata)
    elif mode == '1d':
        xc = _1d_local_centroid1d(xdata, submin=False)
        yc = _1d_local_centroid1d(ydata, submin=False)
    elif mode == '1dsubmin':
        xc = _1d_local_centroid1d(xdata, submin=True)
        yc = _1d_local_centroid1d(ydata, submin=True)

    return (xc + xmax, yc + ymax)
