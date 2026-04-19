#!/usr/bin/env python3
"""
Quick start: generate NHSB waveforms and compare to BBH baseline.

Reproduces the basic waveform diagnostic from §5 of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from nhsb_waveform import NHSBWaveform, NHSBParams, SourceParams

# Frequency grid
f = np.arange(20.0, 512.0, 0.5)

# GW150914-like source
src = SourceParams(m1=35, m2=30, D_L=410)
print(f"Source: {src.m1}+{src.m2} Msun, D_L={src.D_L} Mpc")
print(f"Remnant: M_f={src.M_final:.1f} Msun, chi_f={src.chi_final:.2f}")
print(f"f_peak={src.f_peak:.0f} Hz, f_220={src.f_220:.0f} Hz")

# Three NHSB parameter sets
configs = {
    "Conservative": NHSBParams(1e-8, 0.2, 0.95, 1.0),
    "Moderate":     NHSBParams(1e-5, 1.0, 0.90, 0.7),
    "Strong":       NHSBParams(1e-4, 3.0, 0.80, 0.5),
}

wf = NHSBWaveform()
h_bbh = wf.h_tilde_bbh(f, src)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for label, nhsb in configs.items():
    h = wf.h_tilde(f, src, nhsb)
    dpsi = wf.phase_deformation(f, src, nhsb)
    valid = wf.valid(f, src, nhsb)

    print(f"\n{label}: eps={nhsb.epsilon}, Ls={nhsb.Lambda_star}, "
          f"A={nhsb.A_abs}, d={nhsb.delta} — valid={valid}")

    # Amplitude
    axes[0, 0].loglog(f, np.abs(h) * f, label=label, alpha=0.8)
    # Phase deviation
    axes[0, 1].plot(f, dpsi, label=label, alpha=0.8)
    # Amplitude ratio
    axes[1, 0].plot(f, np.abs(h) / np.abs(h_bbh), label=label, alpha=0.8)

axes[0, 0].loglog(f, np.abs(h_bbh) * f, 'k--', label='BBH', lw=1)
axes[0, 0].set_xlabel('f [Hz]')
axes[0, 0].set_ylabel(r'$|h(f)| \times f$')
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_title('Amplitude spectrum')

axes[0, 1].set_xlabel('f [Hz]')
axes[0, 1].set_ylabel(r'$\Delta\Psi$ [rad]')
axes[0, 1].legend(fontsize=8)
axes[0, 1].set_title('Phase deviation from BBH')

axes[1, 0].axhline(1, color='k', ls='--', lw=0.5)
axes[1, 0].set_xlabel('f [Hz]')
axes[1, 0].set_ylabel(r'$|h_{\rm NHSB}|/|h_{\rm BBH}|$')
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_title('Amplitude ratio')

# Taper windows
from nhsb_waveform.tapers import taper_conservative, taper_heating, taper_ringdown
axes[1, 1].plot(f, taper_conservative(f, src.f_peak), label=r'$W_\Lambda$')
axes[1, 1].plot(f, taper_heating(f, src.f_peak), label=r'$W_A$')
axes[1, 1].plot(f, taper_ringdown(f, src.f_220), label=r'$W_R$')
axes[1, 1].axvline(src.f_peak, color='gray', ls=':', lw=0.5)
axes[1, 1].axvline(src.f_220, color='gray', ls='--', lw=0.5)
axes[1, 1].set_xlabel('f [Hz]')
axes[1, 1].set_ylabel('Window value')
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_title('Taper windows')

plt.suptitle('NHSB Waveform Diagnostic', fontsize=13)
plt.tight_layout()
plt.savefig('nhsb_quickstart.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved nhsb_quickstart.png")
