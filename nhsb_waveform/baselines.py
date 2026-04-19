"""
Baseline BBH waveform generators for the NHSB model.

Three backends are provided:
    ToyInspiralBaseline — lightweight Newtonian-like inspiral
    PyCBCBaseline       — wraps pycbc.waveform (requires pycbc)
    LALSimBaseline      — wraps lalsimulation (requires lalsuite)
    AutoBaseline        — selects best available at runtime
"""

from abc import ABC, abstractmethod
import numpy as np

from nhsb_waveform.params import SourceParams, MTSUN_SI, C_SI, G_SI, MSUN_KG

PI = np.pi


class BaselineWaveform(ABC):
    """Abstract interface for a frequency-domain BBH baseline."""

    @abstractmethod
    def h_tilde(self, f, src):
        """Return complex frequency-domain waveform h~(f).

        Parameters
        ----------
        f : ndarray
            Frequency array in Hz.
        src : SourceParams
            Binary source parameters.

        Returns
        -------
        ndarray (complex)
            Frequency-domain strain.
        """

    @abstractmethod
    def phi_tidal_lib(self, f, src):
        """Leading-order tidal phase basis Φ_tidal^lib(f).

        Scales as v^10 at leading PN order.
        """

    @abstractmethod
    def phi_heat_lib(self, f, src):
        """Leading-order heating phase basis Φ_heat^lib(f).

        Scales as v^8 at leading PN order in the toy version.
        """


class ToyInspiralBaseline(BaselineWaveform):
    """Lightweight Newtonian-like inspiral waveform.

    Uses a stationary-phase-approximation (SPA) inspiral waveform
    with 1PN phase corrections and Newtonian amplitude.  Suitable
    for testing the NHSB deformation logic in isolation.

    This is the baseline used for all synthetic injections and
    Fisher/MCMC studies in the paper.
    """

    def h_tilde(self, f, src):
        amp = self._amplitude(f, src)
        psi = self._phase(f, src)
        return amp * np.exp(1j * psi)

    def _amplitude(self, f, src):
        """Newtonian inspiral amplitude."""
        return (np.sqrt(5 * PI / 24) * src.Mc_seconds ** (5.0 / 6.0)
                / (src.D_L_metres / C_SI) * f ** (-7.0 / 6.0))

    def _phase(self, f, src):
        """SPA phase with 1PN + 1.5PN corrections."""
        v = (PI * src.M * MTSUN_SI * f) ** (1.0 / 3.0)
        return ((3.0 / (128.0 * src.eta)) * v ** (-5)
                * (1.0 + (20.0 / 9.0) * v ** 2 - 16.0 * PI * v ** 3))

    def phi_tidal_lib(self, f, src):
        """Toy tidal phase basis: ~ v^10."""
        v = (PI * src.M * MTSUN_SI * f) ** (1.0 / 3.0)
        return -19.5 * v ** 10

    def phi_heat_lib(self, f, src):
        """Toy heating phase basis: ~ v^8 / (1 + v^2)."""
        v = (PI * src.M * MTSUN_SI * f) ** (1.0 / 3.0)
        return -0.2 * v ** 8 / (1.0 + v ** 2)


class PyCBCBaseline(BaselineWaveform):
    """Adapter wrapping pycbc.waveform.get_fd_waveform().

    Supports any PyCBC-available approximant (e.g., IMRPhenomD,
    IMRPhenomXAS).  Requires the pycbc package.
    """

    def __init__(self, approximant="IMRPhenomD"):
        self.approximant = approximant
        try:
            import pycbc.waveform  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyCBCBaseline requires the pycbc package.  "
                "Install with: pip install pycbc"
            )

    def h_tilde(self, f, src):
        from pycbc.waveform import get_fd_waveform

        hp, _ = get_fd_waveform(
            approximant=self.approximant,
            mass1=src.m1,
            mass2=src.m2,
            spin1z=src.chi1,
            spin2z=src.chi2,
            delta_f=f[1] - f[0],
            f_lower=f[0],
            distance=src.D_L,
            inclination=src.iota,
            coa_phase=src.phi_c,
        )
        # Interpolate onto the requested frequency grid
        hp_data = np.array(hp.data)
        f_pycbc = np.arange(len(hp_data)) * hp.delta_f
        return np.interp(f, f_pycbc, hp_data, left=0, right=0)

    def phi_tidal_lib(self, f, src):
        # Use the toy basis as proxy; production upgrade replaces this
        toy = ToyInspiralBaseline()
        return toy.phi_tidal_lib(f, src)

    def phi_heat_lib(self, f, src):
        toy = ToyInspiralBaseline()
        return toy.phi_heat_lib(f, src)


class LALSimBaseline(BaselineWaveform):
    """Adapter wrapping lalsimulation.SimInspiralChooseFDWaveform().

    Requires the lalsuite package.
    """

    def __init__(self, approximant="IMRPhenomD"):
        self.approximant = approximant
        try:
            import lalsimulation  # noqa: F401
        except ImportError:
            raise ImportError(
                "LALSimBaseline requires lalsuite.  "
                "Install with: pip install lalsuite"
            )

    def h_tilde(self, f, src):
        import lal
        import lalsimulation as lalsim

        m1_kg = src.m1 * MSUN_KG
        m2_kg = src.m2 * MSUN_KG
        d_metres = src.D_L * 3.085677581e22
        df = f[1] - f[0]
        f_lower = float(f[0])
        approx = lalsim.GetApproximantFromString(self.approximant)

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_kg, m2_kg,
            0, 0, src.chi1,
            0, 0, src.chi2,
            d_metres, src.iota, src.phi_c,
            0, 0, 0,
            df, f_lower, f_lower + len(f) * df, f_lower,
            None, approx,
        )
        hp_data = np.array(hp.data.data)
        f_lal = np.arange(len(hp_data)) * df
        return np.interp(f, f_lal, hp_data, left=0, right=0)

    def phi_tidal_lib(self, f, src):
        toy = ToyInspiralBaseline()
        return toy.phi_tidal_lib(f, src)

    def phi_heat_lib(self, f, src):
        toy = ToyInspiralBaseline()
        return toy.phi_heat_lib(f, src)


def AutoBaseline(approximant="IMRPhenomD"):
    """Factory: select the best available baseline at runtime.

    Tries PyCBC first, then LALSim, then falls back to the toy
    inspiral baseline with a warning.

    Parameters
    ----------
    approximant : str
        Waveform approximant name (for PyCBC/LAL backends).

    Returns
    -------
    BaselineWaveform
        The best available baseline instance.
    """
    try:
        return PyCBCBaseline(approximant)
    except ImportError:
        pass
    try:
        return LALSimBaseline(approximant)
    except ImportError:
        pass
    import warnings
    warnings.warn(
        "Neither pycbc nor lalsuite found.  "
        "Using ToyInspiralBaseline (suitable for structural "
        "validation only, not production PE).",
        stacklevel=2,
    )
    return ToyInspiralBaseline()
