#!/usr/bin/env python3
"""
Fisher-matrix detectability forecast.

Reproduces Table V from the paper: σ(Λ★), σ(A), and the ~70×
channel hierarchy across SNR configurations.
"""

import numpy as np
from nhsb_waveform import SourceParams
from nhsb_waveform.fisher import fisher_matrix

f = np.arange(20.0, 512.0, 0.5)
src = SourceParams(m1=35, m2=30, D_L=410)

print("NHSB Fisher Forecast — GW150914-like source")
print("=" * 60)
print(f"{'Configuration':<22} {'σ(Λ★)':>10} {'σ(A)':>10} {'ratio':>8}")
print("-" * 60)

for label, snr in [("aLIGO (SNR~24)", 24),
                   ("A+ (SNR~48)", 48),
                   ("O5 (SNR~100)", 100),
                   ("CE (SNR~1000)", 1000)]:
    _, sigma, _ = fisher_matrix(f, src, snr_target=snr)
    ratio = sigma[0] / sigma[1]
    print(f"{label:<22} {sigma[0]:>10.1f} {sigma[1]:>10.2f} {ratio:>8.0f}×")

print("-" * 60)
print("The ~70× ratio is constant across SNR — this is the")
print("channel hierarchy: heating >> conservative tidal >> echoes.")
