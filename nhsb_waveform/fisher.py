"""
Fisher-matrix detectability forecast for the NHSB model.

Computes the 2×2 Fisher matrix for the two phase-dominant NHSB
parameters (Λ★, A) at fixed source parameters and ε, δ.
See §5.4 of the paper.
"""

import numpy as np
from nhsb_waveform.params import NHSBParams, SourceParams, MTSUN_SI
from nhsb_waveform.baselines import ToyInspiralBaseline
from nhsb_waveform.psd import aLIGO_design_psd
from nhsb_waveform.constitutive import cavity_timescale
from nhsb_waveform.tapers import taper_conservative, taper_heating

PI = np.pi


def fisher_matrix(f, src, nhsb_fiducial=None, Sn=None, df=None,
                  baseline=None, snr_target=None):
    """Compute the 2×2 Fisher matrix for (Λ★, A).

    Parameters
    ----------
    f : ndarray
        Frequency array in Hz.
    src : SourceParams
        Source parameters.
    nhsb_fiducial : NHSBParams, optional
        Fiducial NHSB point for linearisation.
        Defaults to NHSBParams().
    Sn : ndarray, optional
        One-sided PSD.  Defaults to aLIGO design.
    df : float, optional
        Frequency spacing.  Defaults to f[1] - f[0].
    baseline : BaselineWaveform, optional
        Baseline waveform generator.
    snr_target : float, optional
        If given, rescale noise to achieve this target SNR.

    Returns
    -------
    F : ndarray, shape (2, 2)
        Fisher information matrix for (Λ★, A).
    sigma : ndarray, shape (2,)
        1σ measurement uncertainties [σ(Λ★), σ(A)].
    C : ndarray, shape (2, 2)
        Covariance matrix (inverse of F).
    """
    if nhsb_fiducial is None:
        nhsb_fiducial = NHSBParams()
    if baseline is None:
        baseline = ToyInspiralBaseline()
    if Sn is None:
        Sn = aLIGO_design_psd(f)
    if df is None:
        df = f[1] - f[0]

    # Amplitude and velocity
    amp = baseline._amplitude(f, src) if hasattr(baseline, '_amplitude') \
        else np.abs(baseline.h_tilde(f, src))
    v = (PI * src.M * MTSUN_SI * f) ** (1.0 / 3.0)

    # SNR scaling
    snr_baseline = np.sqrt(4 * df * np.sum(amp ** 2 / Sn))
    if snr_target is not None:
        noise_scale = (snr_baseline / snr_target) ** 2
    else:
        noise_scale = 1.0
    Sn_eff = Sn * noise_scale

    # Phase derivatives at fiducial point
    tau_c = cavity_timescale(src.M_final_seconds, nhsb_fiducial.epsilon)

    # d(ΔΨ)/d(Λ★): taper × (1/|ln ε|) × Ξ_δ × Φ_tidal
    from nhsb_waveform.constitutive import microtexture_filter
    Xi = microtexture_filter(f, tau_c, nhsb_fiducial.delta)
    W_L = taper_conservative(f, src.f_peak)
    phi_tidal = baseline.phi_tidal_lib(f, src)
    dpsi_dL = W_L * (1.0 / nhsb_fiducial.abs_ln_epsilon) * Xi * phi_tidal

    # d(ΔΨ)/d(A): Φ_heat (no taper — merger-capable convention)
    # The toy heating basis is used with W_A = 1 in the Fisher
    # forecast to match the paper's convention.  See §3.4: "If
    # the heating basis extends through merger, W_A(f) = 1."
    phi_heat = baseline.phi_heat_lib(f, src)
    dpsi_dA = phi_heat

    # Fisher matrix
    w = amp ** 2 / Sn_eff
    F = np.array([
        [4 * df * np.sum(w * dpsi_dL ** 2),
         4 * df * np.sum(w * dpsi_dL * dpsi_dA)],
        [4 * df * np.sum(w * dpsi_dL * dpsi_dA),
         4 * df * np.sum(w * dpsi_dA ** 2)],
    ])

    C = np.linalg.inv(F)
    sigma = np.sqrt(np.diag(C))
    return F, sigma, C
