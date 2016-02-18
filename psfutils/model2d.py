"""
This module provides discrete fittable models besed on 2D images.
"""
from __future__ import absolute_import, division, unicode_literals,\
     print_function

import numpy as np
from scipy.interpolate import RectBivariateSpline
from astropy.modeling import FittableModel, Fittable2DModel
from astropy.modeling.parameters import Parameter

__all__ = ['Discrete2DModel']


class Discrete2DModel(Fittable2DModel):
    """
    A discrete fittable 2D model of an image.

    This class stores a discrete 2D image and computes the values at arbitrary
    locations (including at intra-pixel, fractional positions) within this
    image using spline interpolation provided by
    :py:class:`~scipy.interpolate.RectBivariateSpline`. Even though this
    particular spline interpolator does not support weighted smoothing,
    :py:class:`Discrete2DModel` can be used to store image weights
    so that these can be passed to the fitter.

    The fittable model provided by this class has three model parameters:
    total flux of the underlying image, and two shifts along each axis of the
    image.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing 2D image.

    origin : tuple, None, optional
        Origin of the coordinate system in image pixels. Origin indicates where
        in the image model coordinates `x` and `y` are zero. If `origin` is
        `None`, then model's origin will be set to the center of the image.

    weights : numpy.ndarray, None, optional
        An array of weights for the corresponding data.

    degree : int, tuple, optional
        Degree of the interpolating spline. A tuple can be used to provide
        different degrees for the X- and Y-axes.

    s : float, optional
        Non-negative smoothing factor. s=0 corresponds to interpolation.
        See :py:class:`~scipy.interpolate.RectBivariateSpline` for more
        details.

    fillval : float, optional
        The value to be returned by the `evaluate` or `__call__` methods
        when evaluation is performed outside the definition domain of the
        model.

    """
    flux = Parameter(description='Total flux of the image.', default=None)
    dx = Parameter(description='Shift along the X-axis relative to the '
                   'origin.', default=0.0)
    dy = Parameter(description='Shift along the Y-axis relative to the '
                   'origin.', default=0.0)

    def __init__(self, data, flux=flux.default, dx=dx.default, dy=dy.default,
                 origin=None, weights=None, degree=3, s=0, fillval=0.0):
        """
        """
        self._fillval = fillval
        self._weights = weights

        # compute flux and normalize data so that sum(data) = 1:
        tflux = np.sum(data)
        if flux is None:
            flux = tflux
        self._ndata = data / tflux

        # set input image related parameters:
        self._ny, self._nx = data.shape

        # find origin of the coordinate system in image's pixel grid:
        self.origin = origin

        super(Discrete2DModel, self).__init__(flux, dx, dy)

        # define interpolating spline:
        self._set_degree(degree)
        self._smoothness = s
        self.recalc_interpolator()

    @property
    def ndata(self):
        """Normalized model data such that sum of all pixels is 1."""
        return self._ndata

    @property
    def data(self):
        """Model data such that sum of all pixels is equal to flux."""
        return (self.flux.value * self._ndata)

    @property
    def weights(self):
        """
        Weights of image data. When setting weights, :py:class:`numpy.ndarray`
        or `None` may be used.
        """
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def shape(self):
        """A tuple of dimensions of the data array in numpy style (ny, nx)."""
        return self._ndata.shape

    @property
    def nx(self):
        """Number of columns in the data array."""
        return self._nx

    @property
    def ny(self):
        """Number of rows in the data array."""
        return self._ny

    @property
    def origin(self):
        """
        A tuple of `x` and `y` coordinates of the origin of the coordinate
        system in terms of pixels of model's image.

        When setting the coordinate system origin, a tuple of two `int` or
        `float` may be used. If origin is set to `None`, the origin of the
        coordinate system will be set to the middle of the data array
        (``(npix-1)/2.0``).

        .. warning::
            Modifying `origin` will not adjust (modify) model's parameters
            `dx` and `dy`.
        """
        return (self._ox, self._oy)

    @origin.setter
    def origin(self, origin):
        if origin is None:
            self._ox = (self._nx - 1) / 2.0
            self._oy = (self._ny - 1) / 2.0
        elif hasattr(origin, '__iter__') and len(origin) == 2:
            self._ox, self._oy = origin
        else:
            raise TypeError("Parameter 'origin' must be either None or an "
                            "iterable with two elements.")

    @property
    def ox(self):
        """X-coordinate of the origin of the coordinate system."""
        return self._ox

    @property
    def oy(self):
        """Y-coordinate of the origin of the coordinate system."""
        return self._oy

    @property
    def degree(self):
        """
        Degree(s) of the interpolating spline: a tuple of two degrees - one for
        each axis.

        When setting the degree, either a single `int` value or a tuple of
        two degrees can be used to set the degree for both axes.

        .. note::
            Use caution when modifying interpolator's degree in a
            computationally intensive part of the code as it may decrease code
            performance due to the need to recompute interpolator.
        """
        return (self._degx, self._degy)

    @degree.setter
    def degree(self, degree):
        self._set_degree(degree)
        self.recalc_interpolator()

    def _set_degree(self, degree):
        if hasattr(degree, '__iter__') and len(degree) == 2:
            self._degx, self._degy = degree
        else:
            self._degx = degree
            self._degy = degree

    @property
    def smoothness(self):
        """
        Smoothness of the interpolating spline. See
        :py:class:`~scipy.interpolate.RectBivariateSpline` for more details.

        .. note::
            Use caution when modifying interpolator's smoothness in a
            computationally intensive part of the code as it may decrease code
            performance due to the need to recompute interpolator.
        """
        return self._smoothness

    @smoothness.setter
    def smoothness(self, s):
        self._smoothness = s
        self.recalc_interpolator()

    @property
    def fillval(self):
        """Fill value to be returned for coordinates outside of the domain of
        definition of the interpolator.
        """
        return self._fillval

    @fillval.setter
    def fillval(self, fillval):
        self._fillval = fillval

    def recenter(self):
        """
        Shift the origin of the coordinate system by amounts indicated by
        the model parameters `dx` and `dy` and set model parameters
        `dx` and `dy` to 0.0.
        """
        self._ox -= self.dx.value
        self._oy -= self.dy.value
        self.dx = 0.0
        self.dy = 0.0

    def recalc_interpolator(self, degree=None, s=None):
        """
        Re-compute the interpolating spline after a change of the spline
        degree or smoothness. Can be used to change the _both_ degree
        _and_ smoothness of the interpolator with a single recomputation
        of the interpolator.

        .. note::
            Use caution when modifying interpolator's degree or smoothness in a
            computationally intensive part of the code as it may decrease code
            performance due to the need to recompute interpolator.
        """
        if degree is not None:
            self._set_degree(degree)

        if s is not None:
            self._smoothness = s

        x = np.arange(self._nx, dtype=np.float)
        y = np.arange(self._ny, dtype=np.float)
        self.psf = RectBivariateSpline(x, y, self._ndata.T,
                                       kx=self._degx, ky=self._degx,
                                       s=self._smoothness)

    def evaluate(self, x, y, flux, dx, dy):
        """
        Evaluate the model on some input variables and provided model
        parameters.
        """
        xi = np.asanyarray(x, dtype=np.float) + (self._ox - dx)
        yi = np.asanyarray(y, dtype=np.float) + (self._oy - dy)

        ipsf = flux * self.psf.ev(xi, yi)

        if self._fillval is not None:
            # find indices of pixels that are outside the input pixel grid and
            # set these pixels to the 'fillval':
            invalid = (((xi < 0) | (xi > self._nx - 1)) |
                       ((yi < 0) | (yi > self._ny - 1)))
            ipsf[invalid] = self._fillval

        return ipsf
