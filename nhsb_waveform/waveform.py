"""
NHSB waveform construction.

Implements the full brane-deformed waveform, Eq. (10):

    h̃_NHSB(f) = h̃_base(f) · exp[i ΔΨ(f)] · [1 + W_R(f) · T(f)]

where ΔΨ is the inspiral phase deformation (tidal + heating)
and T(f) is the weak-cavity transfer function.
"""

import numpy as np

from nhsb_waveform.params import NHSBParams, SourceParams
from nhsb_waveform.constitutive import (
    cavity_timescale,
    effective_tidal_deformability,
    transfer_function,
)
from nhsb_waveform.tapers import (
    taper_conservative,
    taper_heating,
    taper_ringdown,
)
from nhsb_waveform.baselines import BaselineWaveform, ToyInspiralBaseline


class NHSBWaveform:
    """Full NHSB waveform generator.

    Parameters
    ----------
    baseline : BaselineWaveform, optional
        BBH baseline waveform generator.  Defaults to
        ToyInspiralBaseline.

    Examples
    --------
    >>> from nhsb_waveform import NHSBWaveform, NHSBParams, SourceParams
    >>> import numpy as np
    >>> wf = NHSBWaveform()
    >>> f = np.arange(20.0, 512.0, 0.5)
    >>> src = SourceParams(m1=35, m2=30)
    >>> nhsb = NHSBParams(epsilon=1e-8, Lambda_star=0.2,
    ...                   A_abs=0.95, delta=1.0)
    >>> h = wf.h_tilde(f, src, nhsb)
    """

    def __init__(self, baseline=None):
        self.baseline = baseline or ToyInspiralBaseline()

    def h_tilde(self, f, src, nhsb):
        """Compute the full NHSB waveform.  Eq. (10).

        Parameters
        ----------
        f : ndarray
            Frequency array in Hz.
        src : SourceParams
            Binary source parameters.
        nhsb : NHSBParams
            NHSB model parameters.

        Returns
        -------
        ndarray (complex)
            Frequency-domain NHSB waveform.
        """
        # Baseline waveform
        h_base = self.baseline.h_tilde(f, src)

        # Cavity timescale
        tau_c = cavity_timescale(src.M_final_seconds, nhsb.epsilon)

        # Phase deformation: ΔΨ = W_Λ · ΔΨ_Λ + W_A · ΔΨ_A
        delta_psi = self._phase_deformation(f, src, nhsb, tau_c)

        # Ringdown transfer function correction
        T = transfer_function(f, nhsb, tau_c)
        W_R = taper_ringdown(f, src.f_220)

        # Full waveform: Eq. (10)
        return h_base * np.exp(1j * delta_psi) * (1.0 + W_R * T)

    def h_tilde_bbh(self, f, src):
        """BBH baseline waveform (no NHSB deformation).

        Equivalent to h_tilde with NHSBParams(Lambda_star=0, A_abs=1).
        """
        return self.baseline.h_tilde(f, src)

    def phase_deformation(self, f, src, nhsb):
        """Total inspiral phase deformation ΔΨ(f).

        Returns the full tapered phase correction.
        """
        tau_c = cavity_timescale(src.M_final_seconds, nhsb.epsilon)
        return self._phase_deformation(f, src, nhsb, tau_c)

    def _phase_deformation(self, f, src, nhsb, tau_c):
        """Internal: compute ΔΨ with pre-computed τ_c."""
        # Conservative tidal correction: ΔΨ_Λ = Λ_eff(f) · Φ_tidal
        Lambda_eff = effective_tidal_deformability(f, nhsb, tau_c)
        phi_tidal = self.baseline.phi_tidal_lib(f, src)
        W_Lambda = taper_conservative(f, src.f_peak)
        delta_psi_Lambda = W_Lambda * Lambda_eff * phi_tidal

        # Dissipative heating correction: ΔΨ_A = A · Φ_heat
        phi_heat = self.baseline.phi_heat_lib(f, src)
        W_A = taper_heating(f, src.f_peak)
        delta_psi_A = W_A * nhsb.A_abs * phi_heat

        return delta_psi_Lambda + delta_psi_A

    def valid(self, f, src, nhsb):
        """Check hard consistency cuts (§4.2 of the paper).

        Parameters
        ----------
        f : ndarray
            Frequency array in Hz.
        src : SourceParams
            Binary source parameters.
        nhsb : NHSBParams
            NHSB model parameters.

        Returns
        -------
        bool
            True if the parameter point passes all cuts.
        """
        tau_c = cavity_timescale(src.M_final_seconds, nhsb.epsilon)

        # Cut 1: |T(f)| < 1 everywhere
        T = transfer_function(f, nhsb, tau_c)
        if np.max(np.abs(T)) > 1.0:
            return False

        # Cut 2: ergoregion stability
        A_min = ergoregion_floor(src.chi_final)
        if nhsb.A_abs < A_min:
            return False

        # Cut 3: Λ_eff not too large
        Lambda_eff = effective_tidal_deformability(f, nhsb, tau_c)
        if np.max(np.abs(Lambda_eff)) > 100:
            return False

        return True

    def log_prior(self, nhsb):
        """Example shrinkage prior (§4.1 of the paper).

        Implements:
        - Uniform in -log10(ε) over [3, 20]
        - Cauchy(0, 1) on Λ★, truncated at |Λ★| < 10
        - Uniform(0, 1) on A
        - Uniform(0.25, 1.75) on δ

        Parameters
        ----------
        nhsb : NHSBParams
            NHSB model parameters.

        Returns
        -------
        float
            Log-prior density (unnormalised).
        """
        from nhsb_waveform.priors import log_prior
        return log_prior(nhsb)


def ergoregion_floor(chi_f):
    """Spin-dependent ergoregion stability floor for A_abs.

    Piecewise approximation following Maggio et al. (2019).

    Parameters
    ----------
    chi_f : float
        Remnant dimensionless spin.

    Returns
    -------
    float
        Minimum allowed absorptivity.
    """
    if chi_f <= 0.7:
        return 0.003
    elif chi_f <= 0.9:
        return 0.06
    else:
        return 0.60
