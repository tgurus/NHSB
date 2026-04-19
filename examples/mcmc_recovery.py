#!/usr/bin/env python3
"""
MCMC injection-recovery study.

Reproduces Figure 4 from the paper: four injections at CE
golden-event sensitivity (SNR ~ 1000).
"""

import numpy as np
import matplotlib.pyplot as plt
from nhsb_waveform import NHSBParams, SourceParams
from nhsb_waveform.mcmc import run_mcmc

f = np.arange(20.0, 512.0, 0.5)
src = SourceParams(m1=35, m2=30, D_L=410)
SNR = 1000

cases = [
    ("BBH null", NHSBParams(1e-8, 0.0, 1.0, 1.0)),
    ("Strong",   NHSBParams(1e-8, 5.0, 0.5, 1.0)),
    ("Moderate", NHSBParams(1e-8, 3.0, 0.8, 1.0)),
    ("Conservative", NHSBParams(1e-8, 0.2, 0.95, 1.0)),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, (label, nhsb_true) in zip(axes, cases):
    print(f"Running {label}...")
    chain, acc = run_mcmc(f, src, nhsb_true, snr_target=SNR)

    ax.hist2d(chain[::3, 0], chain[::3, 1], bins=40,
              cmap='Blues', density=True, rasterized=True)
    ax.plot(nhsb_true.Lambda_star, nhsb_true.A_abs,
            'r+', ms=14, mew=2.5, zorder=10, label='Injected')
    ax.set_xlabel(r'$\Lambda_\star$')
    if ax == axes[0]:
        ax.set_ylabel(r'$\mathcal{A}$')
    ax.set_title(f"{label}\n"
                 f"($\\Lambda_\\star$={nhsb_true.Lambda_star}, "
                 f"$\\mathcal{{A}}$={nhsb_true.A_abs})", fontsize=8)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc='upper right')

    med_L = np.median(chain[:, 0])
    med_A = np.median(chain[:, 1])
    print(f"  Λ★: {med_L:.2f} ± {np.std(chain[:,0]):.2f}  "
          f"A: {med_A:.3f} ± {np.std(chain[:,1]):.3f}  "
          f"acc: {acc:.0%}")

fig.suptitle(f'NHSB Parameter Recovery: Toy-Baseline MCMC at SNR = {SNR}',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('nhsb_mcmc_recovery.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved nhsb_mcmc_recovery.png")
