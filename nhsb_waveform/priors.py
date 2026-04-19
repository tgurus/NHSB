"""
Prior distributions for NHSB model parameters.

Implements the recommended priors from §4.1 of the paper.
"""

import numpy as np

from nhsb_waveform.params import NHSBParams


def log_prior(nhsb, Lambda_max=10.0, Lambda_scale=1.0):
    """Log-prior for NHSB parameters (unnormalised).

    Parameters
    ----------
    nhsb : NHSBParams
        NHSB model parameters.
    Lambda_max : float
        Hard truncation on |Λ★| (default 10; exploratory: 20).
    Lambda_scale : float
        Cauchy scale for Λ★ prior (default 1.0).

    Returns
    -------
    float
        Log-prior density.  Returns -inf for rejected points.
    """
    # ε: uniform in x_ε = -log10(ε) over [3, 20]
    x_eps = -np.log10(nhsb.epsilon)
    if not (3.0 <= x_eps <= 20.0):
        return -np.inf

    # Λ★: Cauchy(0, Λ_0) truncated at |Λ★| < Lambda_max
    if abs(nhsb.Lambda_star) > Lambda_max:
        return -np.inf
    lp_Lambda = -np.log(1.0 + (nhsb.Lambda_star / Lambda_scale) ** 2)

    # A: uniform on [0, 1]
    if not (0.0 <= nhsb.A_abs <= 1.0):
        return -np.inf

    # δ: uniform on [0.25, 1.75]
    if not (0.25 <= nhsb.delta <= 1.75):
        return -np.inf

    return lp_Lambda


def sample_prior(rng=None, Lambda_max=10.0, Lambda_scale=1.0):
    """Draw a single sample from the NHSB prior.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator.
    Lambda_max : float
        Hard truncation on |Λ★|.
    Lambda_scale : float
        Cauchy scale for Λ★.

    Returns
    -------
    NHSBParams
        A random draw from the prior.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_eps = rng.uniform(3.0, 20.0)
    epsilon = 10.0 ** (-x_eps)

    # Rejection sample from truncated Cauchy
    while True:
        Ls = Lambda_scale * np.tan(
            np.pi * (rng.uniform() - 0.5)
        )
        if abs(Ls) < Lambda_max:
            break

    A_abs = rng.uniform(0.0, 1.0)
    delta = rng.uniform(0.25, 1.75)

    return NHSBParams(
        epsilon=epsilon,
        Lambda_star=Ls,
        A_abs=A_abs,
        delta=delta,
    )
