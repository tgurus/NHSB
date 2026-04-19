"""
Smooth tapering windows for the NHSB waveform model.

Each window confines a deformation channel to its physically
appropriate frequency band.  See §3.4 and Appendix F of the paper.
"""

import numpy as np


def taper_conservative(f, f_peak, f_off_frac=0.80, sigma=0.10):
    """Conservative tidal window W_Λ(f).  Eq. (13).

    Turns off above ~0.8 × f_peak (inspiral-dominated).

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    f_peak : float
        Peak GW frequency in Hz.
    f_off_frac : float
        Turn-off location as fraction of f_peak (default 0.80).
    sigma : float
        Taper width parameter (default 0.10).

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    f_off = f_off_frac * f_peak
    return 0.5 * (1.0 - np.tanh((f - f_off) / (sigma * f_off)))


def taper_heating(f, f_peak, merger_capable=False,
                  f_off_frac=1.0, sigma=0.08):
    """Heating window W_A(f).  Eq. (14).

    If the heating basis extends through merger (e.g., Mukherjee
    et al. 2025), returns unity.  Otherwise, turns off near f_peak.

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    f_peak : float
        Peak GW frequency in Hz.
    merger_capable : bool
        If True, return W_A = 1 everywhere.
    f_off_frac : float
        Turn-off location as fraction of f_peak (default 1.0).
    sigma : float
        Taper width parameter (default 0.08).

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    if merger_capable:
        return np.ones_like(f)
    f_off = f_off_frac * f_peak
    return 0.5 * (1.0 - np.tanh((f - f_off) / (sigma * f_off)))


def taper_ringdown(f, f_220, sigma=0.12):
    """Ringdown transfer window W_R(f).  Eq. (15).

    Turns on near the dominant QNM frequency.

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    f_220 : float
        Dominant (ell=m=2, n=0) QNM frequency in Hz.
    sigma : float
        Taper width parameter (default 0.12).

    Returns
    -------
    ndarray
        Window values in [0, 1].
    """
    return 0.5 * (1.0 + np.tanh((f - f_220) / (sigma * f_220)))
