"""
This module provides tools for source extraction from images and catalogs.
"""
from __future__ import absolute_import, division, unicode_literals, \
     print_function

import warnings
import numpy as np
from astropy.io import fits

from .model2d import Discrete2DModel
from .psf import find_peak

__all__ = ['extract_stars']


def extract_stars(catalog, image, ext, weights=None, wext=0,
                  extract_size=11, minmag=None, maxpixval=None,
                  recenter=None):
    """Extracts subimages centered on stars from a catalog.

    Given an input catalog of source coordinates in an image, the image and
    optionally a weight map, this function extracts small sub-images centered
    on source coordinates in the catalog. These sub-images are re-packaged
    as discrete 2D fittable models.

    Parameters
    ----------
    catalog : numpy.ndarray, str
        A :py:class:`numpy.ndarray` with 2 or 3 columns or a string to a text
        file containing 2 or three columns. First column must contain the ``x``
        coordinate (in pixels) of the sources in the input image and the second
        column must contain the ``y`` coordinates of the sources. The optional
        third coordinate may contain source magnitudes and it is required
        if `minmag` is not `None`.

    image : numpy.ndarray, str
        A 2D :py:class:`numpy.ndarray` or the name of a FITS file containing
        a 2D image.

    ext : int, tuple
        Extension number or a tuple of extension name and extension version
        indicating the extension in the FITS file given by `image` that
        contains the image. This parameter is ignored if `image` is
        a :py:class:`numpy.ndarray`.

    weights : numpy.ndarray, str, None, optional
        A 2D :py:class:`numpy.ndarray` or the name of a FITS file containing
        a 2D image with weights of the corresponding pixels in the
        input `image`.

    wext : int, tuple, optional
        Extension number or a tuple of extension name and extension version
        indicating the extension in the FITS file given by `weights` that
        contains the image of weights. This parameter is ignored if `weights`
        is a :py:class:`numpy.ndarray`.

    extract_size : int, tuple, numpy.ndarray, optional
        Indicates the size of the extraction region for each source. If a
        single number is provided, then a square extraction region with sides
        of length `extract_size` will be used for each source. A tuple can be
        used to indicate a rectangular extraction region with different
        sides along ``X-`` and ``Y-`` axes. Similarly, a 1D vector of the same
        length as the number of sources or a 2D array of size Nx2 can be used
        to indicate a different extraction box size for each source.

    minmag : float, None, optional
        This parameter is used to select sources from the input catalog
        that have magnitudes larger than the value indicated by `minmag`.
        When `minmag` is not `None`, input `catalog` _must_ contain three
        columns. When `minmag` is `None`, no source filtering by magnitude will
        be performed.

    maxpixval : float, None, optional
        This parameter is used to exclude soiurces from the input `catalog`
        whose central pixel is larger than the value of `maxpixval`. This
        can be used to exclude "saturated" stars. When `maxpixval` is `None`,
        no source filtering based on this criteria will be performed.

    recenter : bool, optional
        Indicates that a new source position should be estimated using
        quadratic fitting to pixels around the peak pixel within the extraction
        box. This may be useful if the positions of the sources in the
        `catalog` are not very accurate.

    Returns
    -------
    starlist : list of Discrete2DModel
        A list of :py:class:`Discrete2DModel` objects one for each source
        in the catalog. the origin of each :py:class:`Discrete2DModel` object
        is set to be equal to the center of the star relative to the
        bottom-left corner of the extracted subimage.

    blc : list of tuples
        A list of tuples containing the bottom-left corner coordinates of each
        subimage corrsponding to a star.

    """
    # load data from files:
    if isinstance(image, str):
        image = fits.getdata(image, ext=ext)

    if isinstance(weights, str):
        if wext is None:
            wext = 0
        weights = fits.getdata(weights, ext=wext)

    if isinstance(catalog, str):
        catalog = np.loadtxt(catalog)

    # change the origin of coordinates from 1 (FITS) to 0 (numpy.ndarray):
    image[:, :2] -= 1

    # process extraction region size:
    bsize = np.zeros((catalog.shape[0], 2), dtype=np.int)
    if isinstance(extract_size, np.ndarray):
        if len(extract_size) == 1:
            bsize[:, 0] = extract_size
            bsize[:, 1] = extract_size
        elif len(extract_size == 2):
            bsize[:, :] = extract_size[:, :2]
        else:
            raise ValueError("'extract_size' must be either a 1D vector or a "
                             "2D array of size Nx2.")
    else:
        bsize[:, :] = extract_size

    # filter out stars below minmag:
    if minmag is not None:
        ind = catalog[:, 2] > minmag
        catalog = catalog[ind, :]
        bsize = bsize[ind, :]

    # filter out saturated stars:
    if maxpixval is not None:
        ixy = (np.round(catalog[:, :2] + 0.5).astype(np.int) -
               1).astype(np.intp)
        pv = image[ixy[:, 0], ixy[:, 1]]
        ind = pv < maxpixval
        catalog = catalog[ind, :]
        bsize = bsize[ind, :]

    # extract stars:
    starlist = []
    blc = []
    ny, nx = image.shape
    for (x, y), (w, h) in zip(catalog[:, :2], bsize):
        xc = int(x)
        yc = int(y)
        x1 = max(0, xc - (w - 1) // 2)
        x2 = min(nx, x1 + w)
        y1 = max(0, yc - (h - 1) // 2)
        y2 = min(ny, y1 + h)
        if x1 == x2 or y1 == y2:
            warnings.warn("Source with coordinates ({}, {}) is being ignored "
                          "because there are too few pixels available around "
                          "its center pixel.".format(x, y))
            continue

        cutout = image[y1:y2, x1:x2]

        if recenter:
            cx, cy = find_peak(cutout, x - x1, y - y1, w=3)
            # re-compute extraction box with improved center value:
            xnew = x1 + cx
            ynew = y1 + cy
            xc = int(xnew)
            yc = int(ynew)
            x1 = max(0, xc - (w - 1) // 2)
            x2 = min(nx, x1 + w)
            y1 = max(0, yc - (h - 1) // 2)
            y2 = min(ny, y1 + h)
            if x1 == x2 or y1 == y2:
                warnings.warn("Source with coordinates ({}, {}) is being "
                              "ignored because there are too few pixels "
                              "available around it.".format(x, y))
                continue
            cx = xnew - x1
            cy = ynew - y1
        else:
            cx = x - x1
            cy = y - y1

        model = Discrete2DModel(cutout, weights=weights, origin=(cx, cy))
        starlist.append(model)
        blc.append((x1, y1))

    return starlist, blc
