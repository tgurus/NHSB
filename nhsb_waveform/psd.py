"""
Detector power spectral density models.

Analytic approximations for use in Fisher forecasts and MCMC
studies.  For production analyses, use official PSD files from
the LVK or CE collaborations.
"""

import numpy as np


def aLIGO_design_psd(f):
    """Analytic aLIGO design-sensitivity PSD.

    Based on the LIGO-T1800044 design curve approximation.

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.

    Returns
    -------
    ndarray
        One-sided PSD in Hz^{-1}.
    """
    f = np.asarray(f, dtype=float)
    f0 = 215.0
    S0 = 1e-49
    x = f / f0
    Sn = S0 * np.abs(
        x ** (-4.14)
        - 5.0 * x ** (-2)
        + 111.0 * (1.0 - 0.5 * x ** 2 + 0.25 * x ** 4) / (1.0 + x ** 2)
    )
    return np.clip(Sn, 1e-50, 1e-40)


def CE_psd(f, generation=1):
    """Cosmic Explorer approximate PSD.

    A simple rescaling of the aLIGO PSD by a factor of ~10 in
    amplitude sensitivity (100 in power).  For illustration only.

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    generation : int
        CE generation (1 or 2).

    Returns
    -------
    ndarray
        One-sided PSD in Hz^{-1}.
    """
    scale = {1: 0.01, 2: 0.003}.get(generation, 0.01)
    return scale * aLIGO_design_psd(f)
