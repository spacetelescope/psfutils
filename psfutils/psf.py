"""
This module provides tools for creation and fitting of empirical PSFs (ePSF)
to stars.
"""
from __future__ import absolute_import, division, unicode_literals, \
     print_function

import warnings
import numpy as np
from astropy.convolution import convolve
from astropy.modeling.polynomial import Polynomial2D
from astropy.modeling import fitting

from .model2d import Discrete2DModel

__all__ = ['psf_from_stars', 'fit_stars', 'find_peak', 'iter_build_psf',
           'compute_residuals']

_kernel_quar = np.array(
    [[ 0.041632, -0.080816,  0.078368, -0.080816,  0.041632],
     [-0.080816, -0.019592,  0.200816, -0.019592, -0.080816],
     [ 0.078368,  0.200816,  0.441632,  0.200816,  0.078368],
     [-0.080816, -0.019592,  0.200816, -0.019592, -0.080816],
     [ 0.041632, -0.080816,  0.078368, -0.080816,  0.041632]]
)

_kernel_quad = np.array(
    [[-0.07428311,  0.01142786,  0.03999952,  0.01142786, -0.07428311],
     [ 0.01142786,  0.09714283,  0.12571449,  0.09714283,  0.01142786],
     [ 0.03999952,  0.12571449,  0.15428215,  0.12571449,  0.03999952],
     [ 0.01142786,  0.09714283,  0.12571449,  0.09714283,  0.01142786],
     [-0.07428311,  0.01142786,  0.03999952,  0.01142786, -0.07428311]]
)


def psf_from_stars(stars, shape=None, oversampling=1.0, degree=3,
                   stat='mean', nclip=0, lsig=3.0, usig=3.0, ker=None):
    """
    Register multiple stars into a single oversampled grid to create an ePSF.

    Parameters
    ----------
    stars : list of Discrete2DModel
        A list of :py:class:`~psfutils.model2d.Discrete2DModel` objects
        containing models of the stars that need to be combined into
        a single oversampled PSF. Star registration relies on correct
        coordinates of the centers of stars which must be set in the
        :py:class:`~psfutils.model2d.Discrete2DModel.origin` property or,
        alternatively, using model ``dx`` and ``dy`` parameters.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models.

    oversampling : float, tuple of float, list of (float, tuple of float), optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    degree : int, tuple of int, optional
        The degree of the interpolating spline. If a tuple is provided, the
        first element of the tuple will indicate the degree along ``X`` axis
        and the second element will indicate the degree along ``Y`` axis.

    stat : str {'pmode1', 'pmode2', 'mean', 'median'}, optional
        When multiple stars contribute to the same pixel in the PSF this
        parameter indicates how the value of that pixel in the PSF is computed
        (i.e., which statistics is to be used):

        * 'pmode1' - SEXTRACTOR-like mode estimate based on a
          modified `Pearson's rule <http://en.wikipedia.org/wiki/Nonparametric_skew#Pearson.27s_rule>`_:
          ``2.5*median-1.5*mean``;
        * 'pmode2' - mode estimate based on
          `Pearson's rule <http://en.wikipedia.org/wiki/Nonparametric_skew#Pearson.27s_rule>`_:
          ``3*median-2*mean``;
        * 'mean' - the mean of the distribution of the "good" pixels (after
          clipping);
        * 'median' - the median of the distribution of the "good" pixels;

    nclip : int, optional
        A non-negative number of clipping iterations to use when computing
        the sky value.

    lsig : float, optional
        Lower clipping limit, in sigma, used when computing PSF pixel value.

    usig : float, optional
        Upper clipping limit, in sigma, used when computing PSF pixel value.

    ker : str {'quad', 'quar'}, numpy.ndarray, None, optional
        PSF is to be convolved with the indicated kernel. ``'quad'`` and
        ``'quar'`` built-in kernels have been optimized for oversampling
        factor 4.

    Returns
    -------
    ePSF : Discrete2DModel
        A discrete fittable 2D model of the ePSF with the origin indicating the
        detected center of the PSF.

    """
    oversampx, oversampy = _parse_tuple_pars(oversampling, name='oversampling')

    nmodels = len(stars)

    # if shape is None, find the minimal shape that will include input PSF's
    # data:
    if shape is None:
        x1, x2 = zip(*[(p.dx.value, p.nx + p.dx.value) for p in stars])
        y1, y2 = zip(*[(p.dy.value, p.ny + p.dy.value) for p in stars])
        xmin = min(x1)
        xmax = max(x2)
        ymin = min(y1)
        ymax = max(y2)
        onx = int(np.ceil((xmax - xmin) * oversampx))
        ony = int(np.ceil((ymax - ymin) * oversampy))

        # we prefer odd sized images
        if onx % 2 == 0:
            onx += 1
        if ony % 2 == 0:
            ony += 1
        shape = (ony, onx)

    else:
        (ony, onx) = shape

    # center of the output grid:
    ocx = (onx - 1) / 2.0
    ocy = (ony - 1) / 2.0

    # create output grid:
    xv = (np.arange(onx, dtype=np.float) - ocx) / oversampx
    yv = (np.arange(ony, dtype=np.float) - ocy) / oversampy
    igx, igy = np.meshgrid(xv, yv)

    # allocate "accumulator" array (to store transformed PSFs):
    apsf = np.empty((ony, onx, nmodels), dtype=np.float)
    apsf.fill(np.nan)

    for k, pm in enumerate(stars):
        # backup model and set fillval to numpy.nan:
        p = pm.copy()
        p.fillval = np.nan
        apsf[:, :, k] = p.evaluate(igx, igy, 1, -p.dx.value, -p.dx.value)

    clipped_psf_data = np.empty(shape, dtype=np.float)
    mask = ~np.isnan(apsf)
    for j in range(ony):
        for i in range(onx):
            clipped_psf_data[j, i] = _pixstat(
                apsf[j, i, mask[j, i, :]], stat=stat,
                nclip=nclip, lsig=lsig, usig=usig, default=0.0
            )

    # smooth PSF:
    smoothed_psf_data = _smoothPSF(clipped_psf_data, ker)

    # normalize to unit flux:
    smoothed_psf_data /= np.sum(smoothed_psf_data)

    # recenter the PSF:
    ocx, ocy = find_peak(smoothed_psf_data, w=3)

    # output model:
    ePSF = Discrete2DModel(smoothed_psf_data, flux=1.0, dx=0, dy=0,
                           origin=(ocx, ocy), degree=degree, fillval=0.0)

    return ePSF


def fit_stars(psf, stars, oversampling, flux0=None, fitbox=7,
              fitter=fitting.LevMarLSQFitter, update_flux=False,
              residuals=False):
    """
    Fit a discrete PSF to stars.

    .. note::
        When models in `stars` contain weights, a weighted fit of the PSF to
        the stars will be performed.

    Parameters
    ----------
    psf : Discrete2DModel
        Model of the PSF.

    stars : list of Discrete2DModel
        A list of :py:class:`~psfutils.model2d.Discrete2DModel` objects
        containing models of the stars to which the PSF must be fitted.
        Fitting procedure relies on correct coordinates of the center of the
        PSF and as close as possible to the correct center positions of stars.
        Positions are derived from both the
        :py:class:`~psfutils.model2d.Discrete2DModel.origin` property and from
        model's ``dx`` and ``dy`` parameters.

    oversampling : float, tuple of float, list of (float, tuple of float), optional
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    flux0 : list of float, None, optional
        Initial estimates of the PSF's flux necessary to fit the PSF to the
        stars. Must be  a list of the same length as input `stars`.
        If `None`, initiall guess will be equal to the flux of the star
        to which the PSF is being fitted.

    fitbox : int, tuple of int, None, optional
        The size of the innermost box centered on stars center to be used for
        PSF fitting. This allows using only a small number of central pixels
        of the star for fitting processed thus ignoring wings. A tuple of
        two integers can be used to indicate separate sizes of the fitting
        box for ``X-`` and ``Y-`` axes. When `fitbox` is `None`, the entire
        star's image will be used for fitting.

    fitter : astropy.modeling.fitting.Fitter, optional
        An :py:class:`astropy.modeling.fitting.Fitter` subclassed fitter object.

    update_flux : bool, optional
        Indicates that fluxes of the returned stars should be replaced with
        PSF's fitted fluxes.

    residuals : bool, optional
        Return a list of `numpy.ndarray` of residuals of the fit
        (star - fitted psf).


    Returns
    -------
    fitted_stars : list of Discrete2DModel
        A list of `~psfutils.model2d.Discrete2DModel` of stars with model
        parameters `~psfutils.model2d.Discrete2DModel.dx` and
        `~psfutils.model2d.Discrete2DModel.dy` set to 0 and
        `~psfutils.model2d.Discrete2DModel.origin` will show fitted center of
        the star. If `update_flux` was `True`, the
        `~psfutils.model2d.Discrete2DModel.flux`
        model parameter will contain fitted flux and the original star's
        flux otherwise.

    fitted_fluxes : list of float
        A list of fluxes by which the normalized PSF must be multiplied to fit
        stars.

    fi : list of dict, None
        For some fitters such as `astropy.modeling.fitting.LevMarLSQFitter`
        this list will contain fitting information in the form of a list of
        ``fit_info`` dictionary. If fitters do not support ``fit_info``, `None`
        will be returned.

    res : list of dict, None
        When `residuals` is `True`, this result will contain a list of
        `numpy.ndarray` arrays with residuals of the fit
        (star - fitted psf). If `residuals` was set to `False`, this result
        will be `None`.

    error_status : list of int
        A list of integers showing the error status of the fit. ``0`` indicates
        successful fit. Values of ``1`` or ``2`` will indicate that star's
        center appears to be outside the model's image.

    """
    if hasattr(stars, '__iter__'):
        nstars = len(stars)
    else:
        stars = [stars]
        nstars = 1

    if nstars == 0:
        return []

    if flux0 is None:
        flux0 = [s.flux.value for s in stars]
    else:
        if len(flux0) != nstars:
            raise ValueError("'flux0' must have the same number of elements as "
                             "the number of input stars")

    # analize fitbox:
    minfbx = min([s.nx for s in stars])
    minfby = min([s.ny for s in stars])
    if fitbox is None:
        # use full grid defined by stars' data size:
        fitbox = (minfbx, minfby)

    elif hasattr(fitbox, '__iter__'):
        if len(fitbox) != 2:
            raise ValueError("'fitbox' must be a tuple of two integers, a "
                             "single integer, or None")
        fitbox = (min(minfbx, fitbox[0]), min(minfby, fitbox[0]))
    else:
        fitbox = min(minfbx, minfby, fitbox)
        fitbox = (fitbox, fitbox)

    # make a copy of the original PSF:
    psf = psf.copy()
    psf.recenter()

    # oversampling:
    oversampling = _parse_oversampling(oversampling, nstars)

    # create grid for fitting box (in stars' grid units):
    width, height = fitbox
    width = int(round(width))
    height = int(round(height))
    xv = np.arange(width, dtype=np.float) - (width - 1) // 2
    yv = np.arange(height, dtype=np.float) - (height - 1) // 2
    igx, igy = np.meshgrid(xv, yv)

    # perform fitting for each star:
    fitted_stars = []
    fitted_fluxes = []
    fit = fitter()
    error_status = []
    add_fit_info = hasattr(fit, 'fit_info')
    fi = [] if add_fit_info else None
    res = [] if residuals else None
    for st, f0, ov in zip(stars, flux0, oversampling):
        err = 0
        ovx, ovy = ov

        sxc = st.ox + st.dx.value
        syc = st.oy + st.dy.value

        rxc = int(round(sxc))
        ryc = int(round(syc))

        x1 = rxc - (width - 1) // 2
        x2 = x1 + width
        y1 = ryc - (height - 1) // 2
        y2 = y1 + height

        # check boundaries of the fitting box:
        if x1 < 0:
            i1 = -x1
            x1 = 0
        else:
            i1 = 0
        if x2 > st.nx:
            i2 = width - (x2 - st.nx)
            x2 = st.nx
        else:
            i2 = width
        if y1 < 0:
            j1 = -y1
            y1 = 0
        else:
            j1 = 0
        if y2 > st.ny:
            j2 = height - (y2 - st.ny)
            y2 = st.ny
        else:
            j2 = height

        if rxc < 0 or rxc > (st.nx - 1) or ryc < 0 or ryc > (st.ny - 1):
            # star's center is outside the extraction box
            err = 1
            fit_info = None
            fitted_psf = psf.copy()
            fitted_psf.flux = st.flux.value
            warnings.warn("Source with coordinates ({}, {}) is being ignored "
                          "because its center pixel is outside the image."
                          .format(st.ox, st.oy))

        elif (i2 - i1) < 3 or (j2 - j1) < 3:
            # star's center is too close to the edge of the star's image:
            err = 2
            fit_info = None
            fitted_psf = psf.copy()
            fitted_psf.flux = st.flux.value
            warnings.warn("Source with coordinates ({}, {}) is being ignored "
                          "because there are too few pixels available around "
                          "its center pixel.".format(st.ox, st.oy))

        else:
            # define PSF sampling grid:
            gx = ((rxc - sxc) * ovx) + igx[j1:j2, i1:i2] * ovx
            gy = ((ryc - syc) * ovy) + igy[j1:j2, i1:i2] * ovy

            # initial guess for fitted flux:
            psf.flux = f0

            # fit PSF to the star:
            if st.weights is None:
                # a separate treatment for the case when fitters
                # do not support weights (star's models must not have
                # weights set in such cases)
                fitted_psf = fit(psf, gx, gy, st.data[y1:y2, x1:x2])
            else:
                fitted_psf = fit(psf, gx, gy, st.data[y1:y2, x1:x2],
                                 weights=st.weights[y1:y2, x1:x2])
            if add_fit_info:
                fit_info = fit.fit_info

        # compute correction to the star's position:
        cst = st.copy()
        cst.dx -= fitted_psf.dx.value / ovx
        cst.dy -= fitted_psf.dy.value / ovy
        cst.recenter()

        if residuals:
            # it is important to compute residuals *before* flux of the 'cst'
            # is updated with fitted PSF estimate:
            res.append(_calc_res(fitted_psf, cst, ovx, ovy))

        if update_flux:
            cst.flux = fitted_psf.flux.value
        fitted_stars.append(cst)

        fitted_fluxes.append(fitted_psf.flux.value)

        if add_fit_info:
            fi.append(fit_info)

        error_status.append(err)

    return (fitted_stars, fitted_fluxes, fi, res, error_status)


def iter_build_psf(stars, psf_shape=None, oversampling=1.0, degree=3,
                   stat='mean', nclip=0, lsig=3.0, usig=3.0, ker=None,
                   fitbox=7, fitter=fitting.LevMarLSQFitter, max_iter=10,
                   accuracy=1e-5, residuals=False, update_flux=False):
    """
    Iteratively build the empirical PSF (ePSF) using input stars and then
    improve star position estimates by fitting this ePSF to stars. The process
    is repeated until stop conditions are met.

    This function has same parameters as the ones in `psf_from_stars` and
    `fit_stars`. Below we describe only new parameters.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of PSF build / star fitting iterations to be performed.

    accuracy : float, optional
        Stop iterations when change of stars' centers between two iterations
        is smalled that indicated.


    Returns
    -------

    ePSF : Discrete2DModel
        A discrete fittable 2D model of the ePSF with the origin indicating the
        detected center of the PSF.

    fitted_stars : list of Discrete2DModel
        A list of `~psfutils.model2d.Discrete2DModel` of stars with model
        parameters `~psfutils.model2d.Discrete2DModel.dx` and
        `~psfutils.model2d.Discrete2DModel.dy` set to 0 and
        `~psfutils.model2d.Discrete2DModel.origin` will show fitted center of
        the star. If `update_flux` was `True`,
        the `~psfutils.model2d.Discrete2DModel.flux`
        model parameter will contain fitted flux and the original star's
        flux otherwise.

    fitted_fluxes : list of float
        A list of fluxes by which the normalized PSF must be multiplied to fit
        stars.

    fi : list of dict, None
        For some fitters such as `astropy.modeling.fitting.LevMarLSQFitter`
        this list will contain fitting information in the form of a list of
        ``fit_info`` dictionary. If fitters do not support ``fit_info``, `None`
        will be returned.

    res : list of dict, None
        When `residuals` is `True`, this result will contain a list of
        `numpy.ndarray` arrays with residuals of the fit
        (star - fitted psf). If `residuals` was set to `False`, this result
        will be `None`.

    err : list of int
        A list of integers showing the error status of the fit. ``0`` indicates
        successful fit. Values of ``1`` or ``2`` will indicate that star's
        center appears to be outside the model's image.
        return (psf, stars, fluxes, fi, res, err, eps, niter)

    eps : list of float
        List of delta of positions of stars beween the last estimate and
        previous estimate ``norm(x_i-x_{i-1})``.

    niter : int
        Number of performed iterations.

    """

    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError("'max_iter' must be non-negative")
    if accuracy <= 0.0:
        raise ValueError("'accuracy' must be a positive number")
    acc2 = accuracy**2

    # make a copy of input stars and reset centers:
    stars = [s.copy() for s in stars]
    nstars = len(stars)
    for s in stars:
        s.recenter()

    # create an array of star centers:
    prev_centers = np.asanyarray([s.origin for s in stars], dtype=np.float)

    # initial estimate of fitted PSF flux:
    fluxes = [s.flux.value for s in stars]

    niter = -1
    eps2 = 2.0 * acc2

    while niter <= max_iter and np.amax(eps2) >= acc2:
        niter += 1

        # improved PSF:
        psf = psf_from_stars(
            stars, shape=psf_shape, oversampling=oversampling,
            degree=degree, stat=stat, nclip=nclip,
            lsig=lsig, usig=usig, ker=ker
        )

        # improved fit of PSF to stars:
        stars, fluxes, fi, res, err = fit_stars(
            psf, stars, oversampling=oversampling, flux0=fluxes,
            fitbox=fitbox, fitter=fitter
        )

        # create an array of star centers at this iteration:
        centers = np.asanyarray([s.origin for s in stars], dtype=np.float)

        # check termination criterion:
        dxy = centers - prev_centers
        eps2 = np.sum(dxy * dxy, axis=1)

        prev_centers = centers

    eps = np.sqrt(eps2)

    if residuals:
        res = compute_residuals(
            psf, stars, oversampling, fitted_psf_fluxes=fluxes
        )
    else:
        res = None

    # set the flux of stars to the flux of the fitted PSF
    if update_flux:
        for s, flux in zip(stars, fluxes):
            s.flux = flux

    return (psf, stars, fluxes, fi, res, err, eps, niter)


def _pixstat(data, stat='mean', nclip=0, lsig=3.0, usig=3.0, default=np.nan):
    if nclip > 0:
        if lsig is None or usig is None:
            raise ValueError("When 'nclip' > 0 neither 'lsig' nor 'usig' "
                             "may be None")
    data = np.ravel(data)
    nd, = data.shape

    if nd == 0:
        return default

    m = np.mean(data)

    if nd == 1:
        return m

    need_std = (stat != 'mean' or nclip > 0)
    if need_std:
        s = np.std(data)

    i = np.ones(nd, dtype=np.bool)

    for x in range(nclip):
        m_prev = m
        s_prev = s
        nd_prev = nd

        # sigma clipping:
        lval = m - lsig * s
        uval = m + usig * s
        i = ((data >= lval) & (data <= uval))
        d = data[i]
        nd, = d.shape
        if nd < 1:
            # return statistics based on previous iteration
            break

        m = np.mean(d)
        s = np.std(d)

        if nd == nd_prev:
            # NOTE: we could also add m == m_prev and s == s_prev
            # NOTE: a more rigurous check would be to see that index array 'i'
            #       did not change but that would be too slow and the current
            #       check is very likely good enough.
            break

    if stat == 'mean':
        return m
    elif stat == 'median':
        return np.median(data[i])
    elif stat == 'pmode1':
        return (2.5 * np.median(data[i]) - 1.5 * m)
    elif stat == 'pmode2':
        return (3.0 * np.median(data[i]) - 2.0 * m)
    else:
        raise ValueError("Unsupported 'stat' value")


def _smoothPSF(psf, kernel):
    if kernel is None:
        return psf
    if kernel == 'quad':
        ker = _kernel_quad
    elif kernel == 'quar':
        ker = _kernel_quar
    elif isinstance(kernel, numpy.ndarray):
        ker = kernel
    else:
        raise ValueError("Unsupported kernel")

    spsf = convolve(psf, ker)

    return spsf


def find_peak(image_data, xmax=None, ymax=None, w=3):
    """
    Find location of the peak in an array.

    Parameters
    ----------
    image_data : numpy.ndarray
        Image data.

    xmax : float, None, optional
        Initial guess of the x-coordinate of the peak.

    ymax : float, None, optional
        Initial guess of the x-coordinate of the peak.

    w : int
        Width of the box around the detected peak pixel used for quadratic
        fitting.

    Returns
    -------
    coord : tuple of float
        A pair of coordinates of the peak.

    """
    # check arguments:
    if ((xmax is None and ymax is not None) or (ymax is None and
                                                xmax is not None)):
        raise ValueError("Both 'xmax' and 'ymax' must be either None or not "
                         "None")

    if xmax is None:
        # find index of the pixel having maximum value:
        jmax, imax = np.unravel_index(np.argmax(image_data), image_data.shape)
    else:
        imax = int(xmax)
        imax += int(round(xmax - imax))
        jmax = int(ymax)
        jmax += int(round(ymax - jmax))

    if w * w < 6:
        # we need at least 6 points to fit a 2D polynomial
        if xmax is None:
            return (imax, jmax)
        else:
            return (xmax, ymax)

    # choose a box around maxval pixel of width w:
    ny, nx = image_data.shape
    x1 = max(0, imax - w // 2)
    x2 = min(nx, x1 + w)
    y1 = max(0, jmax - w // 2)
    y2 = min(ny, y1 + w)

    # if peak is at the edge of the box, return integer indices:
    if imax == x1 or imax == x2 or jmax == y1 or jmax == y2:
        return (imax, jmax)

    if (x2 - x1) < w:
        # expand the box:
        if x1 == 0:
            x2 = min(nx, x1 + w)
        if x2 == nx:
            x1 = max(0, x2 - w)
        if y1 == 0:
            y2 = min(ny, y1 + w)
        if y2 == ny:
            y1 = max(0, y2 - w)
        if (x2 - x1) * (y2 - y1) < 6:
            # we need at least 6 points to fit a 2D polynomial
            return (imax, jmax)

    # fit a 2D 2nd degree polynomial to data:
    xi = np.arange(x1, x2)
    yi = np.arange(y1, y2)
    x, y = np.meshgrid(xi, yi)
    pol = Polynomial2D(2)
    fit = fitting.LinearLSQFitter()
    fpol = fit(pol, x, y, image_data[y1:y2, x1:x2])

    # find maximum of the polynomial:
    c01 = fpol.c0_1.value
    c10 = fpol.c1_0.value
    c11 = fpol.c1_1.value
    c02 = fpol.c0_2.value
    c20 = fpol.c2_0.value

    d = 4 * c02 * c20 - c11**2
    if d <= 0 or ((c20 > 0.0 and c02 >= 0.0) or (c20 >= 0.0 and c02 > 0.0)):
        # polynomial is does not have max. return middle of the window:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    xmax = (c01 * c11 - 2.0 * c02 * c10) / d
    ymax = (c10 * c11 - 2.0 * c01 * c20) / d
    coord = (xmax, ymax)

    return coord


def _calc_res(psf, star, ovx, ovy):
    xv = ovx * np.arange(star.nx, dtype=np.float) - ovx * star.ox
    yv = ovy * np.arange(star.ny, dtype=np.float) - ovy * star.oy
    igx, igy = np.meshgrid(xv, yv)
    psf_star = psf(igx, igy)
    return (star.data - psf_star)


def compute_residuals(psf, stars, oversampling, fitted_psf_fluxes=None):
    """
    Register the `psf` to intput `stars` and compute the difference.

    Parameters
    ----------
    psf : Discrete2DModel
        Model of the PSF.

    stars : list of Discrete2DModel
        A list of :py:class:`~psfutils.model2d.Discrete2DModel` objects
        containing models of the stars.

    shape : tuple, optional
        Numpy-style shape of the output PSF. If shape is not specified (i.e.,
        it is set to `None`), the shape will be derived from the sizes of the
        input star models.

    oversampling : float, tuple of float, list of (float, tuple of float)
        Oversampling factor of the PSF relative to star. It indicates how many
        times a pixel in the star's image should be sampled when creating PSF.
        If a single number is provided, that value will be used for both
        ``X`` and ``Y`` axes and for all stars. When a tuple is provided,
        first number will indicate oversampling along the ``X`` axis and the
        second number will indicate oversampling along the ``Y`` axis. It is
        also possible to have individualized oversampling factors for each star
        by providing a list of integers or tuples of integers.

    fitted_psf_fluxes : list of float, None, optional
        Fluxes that should be used to scale the input `psf`. The list must be
        of the same length as the length of `stars`. If not provided,
        flux of the stars will be used for scaling the PSF.

    Returns
    -------
    res : list of numpy.ndarray
        A list of `numpy.ndarray` of residuals.

    """
    nstars = len(stars)
    oversampling = _parse_oversampling(oversampling, nstars)
    psf = psf.copy()
    res = []
    if fitted_psf_fluxes is None:
        fitted_psf_fluxes = [s.flux.value for s in stars]
    for s, (ovx, ovy), flux in zip(stars, oversampling, fitted_psf_fluxes):
        psf.flux = flux
        res.append(_calc_res(psf, s, ovx, ovy))
    return res


def _parse_tuple_pars(par, default=None, name=''):
    if par is None:
        par = default

    if hasattr(par, '__iter__'):
        if len(par) != 2:
            raise TypeError("Parameter '{:s}' must be either a scalar or an "
                            "iterable with two elements.".format(name))
        px = par[0]
        py = par[1]
    elif par is None:
        return None
    else:
        px = par
        py = par

    return (px, py)


def _parse_oversampling(oversampling, nstars):
    if hasattr(oversampling, '__iter__'):
        if len(oversampling != nstars):
            raise ValueError("The number of oversampling values must be equal "
                             "to the number of stars")
        oversampling = [_parse_tuple_pars(o, name='oversampling')
                        for o in oversampling]
    else:
        oversampling = nstars * \
            [_parse_tuple_pars(oversampling, name='oversampling')]

    return oversampling
