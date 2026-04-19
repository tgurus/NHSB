"""
nhsb_waveform — Near-Horizon Scrambling Brane waveform model.

A four-parameter phenomenological waveform deformation for
dissipative horizonless compact objects.

Reference:
    Reimer Morales, J. (2026). "Near-Horizon Scrambling Brane:
    A Minimal Phenomenological Waveform Model for Dissipative
    Horizonless Compact Objects." Physical Review D (submitted).

Parameters:
    epsilon (ε)      — compactness offset, r_m = r_+(1 + ε)
    Lambda_star (Λ★) — conservative tidal response amplitude
    A_abs (A)        — brane absorptivity ∈ [0, 1]
    delta (δ)        — scrambling exponent

Observational channels:
    1. Conservative tidal dephasing (inspiral)
    2. Dissipative heating (inspiral–merger)
    3. Modified ringdown via weak-cavity transfer function
"""

from nhsb_waveform.params import NHSBParams, SourceParams
from nhsb_waveform.constitutive import (
    cavity_timescale,
    microtexture_filter,
    effective_tidal_deformability,
    reflectivity,
    transfer_function,
)
from nhsb_waveform.tapers import taper_conservative, taper_heating, taper_ringdown
from nhsb_waveform.waveform import NHSBWaveform
from nhsb_waveform.baselines import (
    ToyInspiralBaseline,
    AutoBaseline,
)
from nhsb_waveform.priors import log_prior, sample_prior
from nhsb_waveform.psd import aLIGO_design_psd, CE_psd

__version__ = "1.3.0"

__all__ = [
    "NHSBParams",
    "SourceParams",
    "NHSBWaveform",
    "ToyInspiralBaseline",
    "AutoBaseline",
    "cavity_timescale",
    "microtexture_filter",
    "effective_tidal_deformability",
    "reflectivity",
    "transfer_function",
    "taper_conservative",
    "taper_heating",
    "taper_ringdown",
    "log_prior",
    "sample_prior",
    "aLIGO_design_psd",
    "CE_psd",
]
