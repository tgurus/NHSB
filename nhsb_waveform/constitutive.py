"""
NHSB constitutive laws.

Implements the three toy-model closure relations from the paper:
    Eq. (16): Λ_eff(f)  — effective tidal deformability
    Eq. (19): R(f)      — brane reflectivity
    Eq. (15): τ_c       — cavity timescale

These are phenomenologically motivated ansätze, not derived from
first principles.  See §3.5.1 (Design Criteria) and Appendix E
(Non-Uniqueness) of the paper for discussion.
"""

import numpy as np

PI2 = 2.0 * np.pi


def cavity_timescale(M_final_sec, epsilon, kappa_tau=4.0):
    """Cavity light-crossing timescale.  Eq. (15).

    τ_c = κ_τ · M_f · |ln ε|

    Parameters
    ----------
    M_final_sec : float
        Remnant mass in seconds.
    epsilon : float
        Compactness offset.
    kappa_tau : float
        O(1) prefactor (default 4).

    Returns
    -------
    float
        Cavity timescale in seconds.
    """
    return kappa_tau * M_final_sec * abs(np.log(epsilon))


def microtexture_filter(f, tau_c, delta):
    """Microtexture filter Ξ_δ(f).  Part of Eq. (16).

    Ξ_δ(f) = 1 / [1 + (2π f τ_c)^δ]

    At low f (f ≪ 1/τ_c): Ξ → 1  (quasistatic response).
    At high f (f ≫ 1/τ_c): Ξ → 0  (scrambling suppresses coherence).

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    tau_c : float
        Cavity timescale in seconds.
    delta : float
        Scrambling exponent.

    Returns
    -------
    ndarray
        Microtexture filter values.
    """
    return 1.0 / (1.0 + (PI2 * f * tau_c) ** delta)


def effective_tidal_deformability(f, nhsb, tau_c):
    """Effective tidal deformability Λ_eff(f).  Eq. (16).

    Λ_eff(f) = (Λ★ / |ln ε|) · Ξ_δ(f)

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    nhsb : NHSBParams
        NHSB model parameters.
    tau_c : float
        Cavity timescale in seconds.

    Returns
    -------
    ndarray
        Effective tidal deformability at each frequency.
    """
    Xi = microtexture_filter(f, tau_c, nhsb.delta)
    return (nhsb.Lambda_star / nhsb.abs_ln_epsilon) * Xi


def reflectivity(f, nhsb, tau_c):
    """Brane reflectivity R(f).  Eq. (19).

    R(f) = √(1 - A) · exp[−(2π f τ_c)^δ]

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    nhsb : NHSBParams
        NHSB model parameters.
    tau_c : float
        Cavity timescale in seconds.

    Returns
    -------
    ndarray
        Reflectivity at each frequency.
    """
    prefactor = np.sqrt(max(0.0, 1.0 - nhsb.A_abs))
    return prefactor * np.exp(-((PI2 * f * tau_c) ** nhsb.delta))


def transfer_function(f, nhsb, tau_c, phi0=0.0):
    """Weak-cavity transfer function T(f).  Eq. (17).

    T(f) = R(f) · exp[i(2π f τ_c + φ₀)]
           / (1 − R(f) · exp[i(2π f τ_c + φ₀)])

    Parameters
    ----------
    f : array_like
        Frequency array in Hz.
    nhsb : NHSBParams
        NHSB model parameters.
    tau_c : float
        Cavity timescale in seconds.
    phi0 : float
        Phase offset (default 0).

    Returns
    -------
    ndarray (complex)
        Transfer function values.
    """
    R = reflectivity(f, nhsb, tau_c)
    phase = PI2 * f * tau_c + phi0
    z = R * np.exp(1j * phase)
    # Guard against division by zero (should not happen for |R| < 1)
    denom = 1.0 - z
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    return z / denom
