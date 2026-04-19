"""
Parameter containers for the NHSB waveform model.

Implements the four NHSB model parameters (Table 1 of the paper)
and source parameters for binary systems.
"""

from dataclasses import dataclass, field
import numpy as np

# Physical constants
MTSUN_SI = 4.925491025543576e-6  # M_sun in seconds
G_SI = 6.67430e-11               # m^3 kg^-1 s^-2
C_SI = 299792458.0               # m/s
MSUN_KG = 1.98892e30             # kg
MPC_SI = 3.085677581e22          # metres per Mpc


@dataclass
class NHSBParams:
    """Four-parameter NHSB model specification.

    Parameters
    ----------
    epsilon : float
        Compactness offset.  r_m = r_+(1 + epsilon).
        Typical range: 1e-20 to 1e-3.
    Lambda_star : float
        Conservative tidal response amplitude.
        Default prior: |Lambda_star| < 10 (recommended),
        |Lambda_star| < 20 (exploratory).
    A_abs : float
        Brane absorptivity, in [0, 1].
        A_abs = 1 recovers the BH absorption limit.
    delta : float
        Scrambling exponent controlling the microtexture filter.
        delta = 1: Debye-like (Markovian) relaxation.
        delta < 1: stretched-exponential (glassy/hierarchical).
        delta > 1: sharper UV suppression.
    """

    epsilon: float = 1e-8
    Lambda_star: float = 0.0
    A_abs: float = 1.0
    delta: float = 1.0

    def __post_init__(self):
        if not 0 < self.epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")
        if not 0 <= self.A_abs <= 1:
            raise ValueError(f"A_abs must be in [0, 1], got {self.A_abs}")
        if self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")

    @property
    def log_epsilon(self):
        """Natural logarithm of epsilon (negative)."""
        return np.log(self.epsilon)

    @property
    def abs_ln_epsilon(self):
        """|ln(epsilon)|, the compactness suppression factor."""
        return abs(np.log(self.epsilon))


@dataclass
class SourceParams:
    """Binary compact-object source parameters.

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    chi1, chi2 : float
        Dimensionless spin magnitudes (aligned-spin only).
    D_L : float
        Luminosity distance in Mpc.
    iota : float
        Inclination angle in radians.
    phi_c : float
        Coalescence phase in radians.
    """

    m1: float = 35.0
    m2: float = 30.0
    chi1: float = 0.0
    chi2: float = 0.0
    D_L: float = 410.0
    iota: float = 0.0
    phi_c: float = 0.0

    def __post_init__(self):
        if self.m1 < self.m2:
            self.m1, self.m2 = self.m2, self.m1

    @property
    def M(self):
        """Total mass in solar masses."""
        return self.m1 + self.m2

    @property
    def eta(self):
        """Symmetric mass ratio."""
        return self.m1 * self.m2 / self.M**2

    @property
    def Mc(self):
        """Chirp mass in solar masses."""
        return self.M * self.eta**0.6

    @property
    def Mc_seconds(self):
        """Chirp mass in seconds (geometric units)."""
        return G_SI * self.Mc * MSUN_KG / C_SI**3

    @property
    def M_final(self):
        """Remnant mass estimate (simple fit)."""
        return self.M * (1 - 0.04 * self.eta)

    @property
    def M_final_seconds(self):
        """Remnant mass in seconds."""
        return self.M_final * MTSUN_SI

    @property
    def chi_final(self):
        """Remnant spin estimate (simple fit)."""
        return min(0.69, 0.69 * (1 - 0.1 * (1 - 4 * self.eta)))

    @property
    def f_peak(self):
        """Peak GW frequency in Hz."""
        return 0.08 / self.M_final_seconds

    @property
    def f_220(self):
        """Dominant (ell=m=2, n=0) QNM frequency in Hz."""
        return (0.055 + 0.03 * self.chi_final) / self.M_final_seconds

    @property
    def D_L_metres(self):
        """Luminosity distance in metres."""
        return self.D_L * MPC_SI
