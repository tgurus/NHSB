"""
Metropolis-Hastings MCMC for NHSB parameter recovery.

Implements the injection-recovery study from §5.5 of the paper.
Samples over (Λ★, A) with fixed source parameters and (ε, δ).
"""

import numpy as np

from nhsb_waveform.params import NHSBParams, SourceParams, MTSUN_SI
from nhsb_waveform.baselines import ToyInspiralBaseline
from nhsb_waveform.constitutive import cavity_timescale
from nhsb_waveform.tapers import taper_conservative, taper_heating
from nhsb_waveform.psd import aLIGO_design_psd

PI = np.pi


def run_mcmc(f, src, nhsb_true, Sn=None, snr_target=1000,
             n_samples=120000, n_burn=30000,
             proposal_sigma=None, seed=42,
             Lambda_max=10.0, baseline=None):
    """Run Metropolis-Hastings MCMC over (Λ★, A).

    Parameters
    ----------
    f : ndarray
        Frequency array in Hz.
    src : SourceParams
        Source parameters (fixed during sampling).
    nhsb_true : NHSBParams
        Injected (true) NHSB parameters.
    Sn : ndarray, optional
        One-sided PSD.  Defaults to aLIGO design.
    snr_target : float
        Target SNR (noise is rescaled).
    n_samples : int
        Total number of MCMC samples.
    n_burn : int
        Number of burn-in samples to discard.
    proposal_sigma : ndarray, shape (2,), optional
        Proposal widths for (Λ★, A).  Default: [1.5, 0.1].
    seed : int
        Random seed for reproducibility.
    Lambda_max : float
        Hard prior truncation on |Λ★|.
    baseline : BaselineWaveform, optional
        Waveform baseline.

    Returns
    -------
    chain : ndarray, shape (n_samples - n_burn, 2)
        Posterior samples (Λ★, A) after burn-in.
    acceptance_rate : float
        Fraction of accepted proposals.
    """
    if baseline is None:
        baseline = ToyInspiralBaseline()
    if Sn is None:
        Sn = aLIGO_design_psd(f)
    if proposal_sigma is None:
        proposal_sigma = np.array([1.5, 0.1])

    df = f[1] - f[0]
    nf = len(f)

    # Compute amplitude and phase ingredients
    amp = baseline._amplitude(f, src) if hasattr(baseline, '_amplitude') \
        else np.abs(baseline.h_tilde(f, src))
    v = (PI * src.M * MTSUN_SI * f) ** (1.0 / 3.0)
    psi_base = baseline._phase(f, src) if hasattr(baseline, '_phase') \
        else np.angle(baseline.h_tilde(f, src))

    # Noise scaling
    snr_baseline = np.sqrt(4 * df * np.sum(amp ** 2 / Sn))
    noise_scale = snr_baseline / snr_target
    Sn_eff = Sn * noise_scale ** 2

    tau_c = cavity_timescale(src.M_final_seconds, nhsb_true.epsilon)

    def nhsb_phase(Ls, A):
        from nhsb_waveform.constitutive import microtexture_filter
        Xi = microtexture_filter(f, tau_c, nhsb_true.delta)
        W_L = taper_conservative(f, src.f_peak)
        lam_eff = Ls / nhsb_true.abs_ln_epsilon * Xi
        phi_tidal = baseline.phi_tidal_lib(f, src)
        phi_heat = baseline.phi_heat_lib(f, src)
        W_A = taper_heating(f, src.f_peak)
        return W_L * lam_eff * phi_tidal + W_A * A * phi_heat

    # Generate injection + noise
    dpsi_true = nhsb_phase(nhsb_true.Lambda_star, nhsb_true.A_abs)
    h_signal = amp * np.exp(1j * (psi_base + dpsi_true))
    rng = np.random.default_rng(seed)
    noise = (rng.normal(0, np.sqrt(Sn_eff / (4 * df)), nf)
             + 1j * rng.normal(0, np.sqrt(Sn_eff / (4 * df)), nf))
    h_data = h_signal + noise

    # Log-likelihood
    def log_like(Ls, A):
        dpsi = nhsb_phase(Ls, A)
        h_model = amp * np.exp(1j * (psi_base + dpsi))
        diff = h_data - h_model
        return -2.0 * df * np.sum(np.abs(diff) ** 2 / Sn_eff)

    # MCMC
    chain = np.zeros((n_samples, 2))
    chain[0] = [0.0, 1.0]  # Start at BBH
    ll_current = log_like(0.0, 1.0)
    accepts = 0

    for i in range(1, n_samples):
        prop = chain[i - 1] + rng.normal(0, proposal_sigma)

        # Prior bounds
        if prop[1] < 0 or prop[1] > 1 or abs(prop[0]) > Lambda_max:
            chain[i] = chain[i - 1]
            continue

        # Cauchy prior ratio
        log_prior_ratio = (np.log(1 + chain[i - 1, 0] ** 2)
                           - np.log(1 + prop[0] ** 2))

        ll_prop = log_like(prop[0], prop[1])
        log_alpha = ll_prop - ll_current + log_prior_ratio

        if np.log(rng.uniform()) < log_alpha:
            chain[i] = prop
            ll_current = ll_prop
            accepts += 1
        else:
            chain[i] = chain[i - 1]

    return chain[n_burn:], accepts / n_samples
