# nhsb-waveform

**Near-Horizon Scrambling Brane: A phenomenological gravitational waveform model for dissipative horizonless compact objects.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

## Overview

`nhsb-waveform` implements the four-parameter NHSB waveform model described in:

> Reimer Morales, J. (2026). "Near-Horizon Scrambling Brane: A Minimal Phenomenological Waveform Model for Dissipative Horizonless Compact Objects." *Physical Review D* (submitted).

The model deforms standard binary black hole (BBH) waveform templates through three observational channels:

1. **Conservative tidal dephasing** during inspiral
2. **Dissipative heating** through merger
3. **Modified ringdown** via a weak-cavity transfer function

The central result is a channel hierarchy: in the almost-BH regime, dissipative heating constrains brane deviations roughly **70× more tightly** than conservative tidal response, with late-time recycled signal (echoes) strongly suppressed.

## Parameters

| Symbol | Name | Range | Physical role |
|--------|------|-------|---------------|
| ε | Compactness offset | 10⁻²⁰ – 10⁻³ | How close the brane sits to the would-be horizon |
| Λ★ | Conservative response | \|Λ★\| < 10 | Tidal deformability amplitude |
| A | Absorptivity | [0, 1] | Brane absorption efficiency (A = 1 → BH limit) |
| δ | Scrambling exponent | [0.25, 1.75] | Microtexture spectral index |

## Installation

```bash
# From source
git clone https://github.com/globalharmonics/nhsb-waveform.git
cd nhsb-waveform
pip install -e .

# With PyCBC backend support
pip install -e ".[pycbc]"

# With LALSuite backend support
pip install -e ".[lal]"
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.21, SciPy ≥ 1.7, Matplotlib ≥ 3.5

## Quick Start

```python
import numpy as np
from nhsb_waveform import NHSBWaveform, NHSBParams, SourceParams

# Set up frequency grid
f = np.arange(20.0, 512.0, 0.5)

# Define source (GW150914-like)
src = SourceParams(m1=35, m2=30, D_L=410)

# Define NHSB parameters (conservative corner)
nhsb = NHSBParams(epsilon=1e-8, Lambda_star=0.2, A_abs=0.95, delta=1.0)

# Generate waveform
wf = NHSBWaveform()
h_nhsb = wf.h_tilde(f, src, nhsb)
h_bbh = wf.h_tilde_bbh(f, src)

# Check consistency cuts
print(f"Valid: {wf.valid(f, src, nhsb)}")

# Phase deviation
delta_psi = wf.phase_deformation(f, src, nhsb)
```

## Fisher Forecast

```python
from nhsb_waveform.fisher import fisher_matrix

F, sigma, C = fisher_matrix(f, src, snr_target=100)
print(f"σ(Λ★) = {sigma[0]:.1f}")
print(f"σ(A)  = {sigma[1]:.2f}")
print(f"Ratio = {sigma[0]/sigma[1]:.0f}×")
```

## MCMC Injection Recovery

```python
from nhsb_waveform.mcmc import run_mcmc

nhsb_true = NHSBParams(epsilon=1e-8, Lambda_star=3.0, A_abs=0.8, delta=1.0)
chain, acc = run_mcmc(f, src, nhsb_true, snr_target=1000)
print(f"Acceptance: {acc:.0%}")
print(f"Λ★: {np.median(chain[:,0]):.2f} ± {np.std(chain[:,0]):.2f}")
print(f"A:  {np.median(chain[:,1]):.3f} ± {np.std(chain[:,1]):.3f}")
```

## Waveform Baselines

Three backends are provided:

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `ToyInspiralBaseline` | Newtonian SPA inspiral (default) | None |
| `PyCBCBaseline` | Wraps `pycbc.waveform` | `pip install pycbc` |
| `LALSimBaseline` | Wraps `lalsimulation` | `pip install lalsuite` |

`AutoBaseline()` selects the best available backend at runtime.

```python
from nhsb_waveform import AutoBaseline, NHSBWaveform

wf = NHSBWaveform(baseline=AutoBaseline("IMRPhenomXAS"))
```

**Note:** The tidal and heating phase bases currently use proxy PN-like functions. Replacing these with calibrated library-backed bases (e.g., the merger-capable heating approximant of Mukherjee et al. 2025) is the primary recommended upgrade for production use.

## Constitutive Laws

The three toy-model closure relations (§3.5 of the paper):

**Effective tidal deformability** (Eq. 16):
```
Λ_eff(f) = (Λ★ / |ln ε|) · 1 / [1 + (2πf τ_c)^δ]
```

**Brane reflectivity** (Eq. 19):
```
R(f) = √(1 − A) · exp[−(2πf τ_c)^δ]
```

**Cavity timescale** (Eq. 15):
```
τ_c = 4 · M_f · |ln ε|
```

These are phenomenologically motivated ansätze, not derived from first principles. See §3.5.1 (Design Criteria) and Appendix E (Non-Uniqueness) of the paper.

## Project Structure

```
nhsb-waveform/
├── nhsb_waveform/
│   ├── __init__.py        # Package API
│   ├── params.py          # NHSBParams, SourceParams dataclasses
│   ├── constitutive.py    # τ_c, Ξ_δ, Λ_eff, R, T (Eqs. 15–19)
│   ├── tapers.py          # W_Λ, W_A, W_R taper windows (Eqs. 13–15)
│   ├── baselines.py       # Toy, PyCBC, LALSim, Auto baselines
│   ├── waveform.py        # NHSBWaveform class
│   ├── priors.py          # Prior distributions (§4.1)
│   ├── psd.py             # Detector PSD models
│   ├── fisher.py          # Fisher matrix forecast (§5.4)
│   └── mcmc.py            # MCMC injection recovery (§5.5)
├── examples/
│   ├── quickstart.py      # Basic waveform generation
│   ├── fisher_forecast.py # Reproduce Table V from the paper
│   ├── mcmc_recovery.py   # Reproduce Figure 4 from the paper
│   └── generate_paper_figures.py  # All paper figures
├── tests/
│   └── test_basic.py      # Unit tests
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ReimerMorales2026NHSB,
    author  = {Reimer Morales, John},
    title   = {Near-Horizon Scrambling Brane: A Minimal Phenomenological
               Waveform Model for Dissipative Horizonless Compact Objects},
    journal = {Physical Review D},
    year    = {2026},
    note    = {Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE).
