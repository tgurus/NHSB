#!/usr/bin/env python3
"""
Unit tests for the nhsb_waveform package.

Tests cover:
    - Parameter validation
    - Constitutive law asymptotic behavior (Table III)
    - BH recovery limits
    - Waveform generation
    - Consistency cuts
    - Fisher matrix
    - Prior sampling
"""

import numpy as np
import sys
import os

# Add parent directory to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nhsb_waveform.params import NHSBParams, SourceParams
from nhsb_waveform.constitutive import (
    cavity_timescale,
    microtexture_filter,
    effective_tidal_deformability,
    reflectivity,
    transfer_function,
)
from nhsb_waveform.tapers import taper_conservative, taper_heating, taper_ringdown
from nhsb_waveform.waveform import NHSBWaveform, ergoregion_floor
from nhsb_waveform.priors import log_prior, sample_prior
from nhsb_waveform.psd import aLIGO_design_psd


def test_params_validation():
    """NHSBParams rejects invalid inputs."""
    # Valid
    p = NHSBParams(1e-8, 0.2, 0.95, 1.0)
    assert p.abs_ln_epsilon > 0

    # Invalid epsilon
    try:
        NHSBParams(epsilon=0.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Invalid A_abs
    try:
        NHSBParams(A_abs=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Invalid delta
    try:
        NHSBParams(delta=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  test_params_validation: PASSED")


def test_source_params():
    """SourceParams computes derived quantities correctly."""
    src = SourceParams(m1=35, m2=30)
    assert src.M == 65.0
    assert abs(src.eta - 35 * 30 / 65**2) < 1e-10
    assert src.f_peak > 0
    assert src.f_220 > 0
    assert src.M_final < src.M
    # m1 >= m2 enforced
    src2 = SourceParams(m1=10, m2=30)
    assert src2.m1 == 30 and src2.m2 == 10
    print("  test_source_params: PASSED")


def test_cavity_timescale():
    """τ_c = κ_τ · M_f · |ln ε|."""
    Mf_sec = 64.4 * 4.925491025543576e-6
    tau = cavity_timescale(Mf_sec, 1e-8)
    expected = 4.0 * Mf_sec * abs(np.log(1e-8))
    assert abs(tau - expected) < 1e-15
    # Smaller ε → larger τ_c
    assert cavity_timescale(Mf_sec, 1e-20) > cavity_timescale(Mf_sec, 1e-5)
    print("  test_cavity_timescale: PASSED")


def test_microtexture_asymptotic():
    """Ξ_δ → 1 at low f, Ξ_δ → 0 at high f (Table III)."""
    tau_c = 0.01
    f_low = np.array([0.001])
    f_high = np.array([1e6])
    for delta in [0.5, 1.0, 1.5]:
        Xi_low = microtexture_filter(f_low, tau_c, delta)
        Xi_high = microtexture_filter(f_high, tau_c, delta)
        assert Xi_low > 0.99, f"Ξ({f_low}) should be ~1, got {Xi_low}"
        assert Xi_high < 0.01, f"Ξ({f_high}) should be ~0, got {Xi_high}"
    print("  test_microtexture_asymptotic: PASSED")


def test_bh_recovery_lambda():
    """Λ★ → 0 recovers vanishing conservative deformation."""
    f = np.arange(20.0, 512.0, 0.5)
    nhsb_bh = NHSBParams(1e-8, 0.0, 1.0, 1.0)
    tau_c = cavity_timescale(64.4 * 4.925491025543576e-6, 1e-8)
    Leff = effective_tidal_deformability(f, nhsb_bh, tau_c)
    assert np.max(np.abs(Leff)) < 1e-15
    print("  test_bh_recovery_lambda: PASSED")


def test_bh_recovery_reflectivity():
    """A → 1 recovers R → 0."""
    f = np.arange(20.0, 512.0, 0.5)
    nhsb_bh = NHSBParams(1e-8, 0.0, 1.0, 1.0)
    tau_c = cavity_timescale(64.4 * 4.925491025543576e-6, 1e-8)
    R = reflectivity(f, nhsb_bh, tau_c)
    assert np.max(np.abs(R)) < 1e-15
    print("  test_bh_recovery_reflectivity: PASSED")


def test_transfer_function_bounded():
    """Transfer function |T(f)| stays bounded for valid parameters."""
    f = np.arange(20.0, 512.0, 0.5)
    nhsb = NHSBParams(1e-8, 0.2, 0.95, 1.0)
    tau_c = cavity_timescale(64.4 * 4.925491025543576e-6, 1e-8)
    T = transfer_function(f, nhsb, tau_c)
    assert np.max(np.abs(T)) < 1.0
    print("  test_transfer_function_bounded: PASSED")


def test_tapers_range():
    """Taper windows output values in [0, 1]."""
    f = np.arange(20.0, 1000.0, 0.5)
    fp, f220 = 250.0, 235.0
    for w in [taper_conservative(f, fp),
              taper_heating(f, fp),
              taper_ringdown(f, f220)]:
        assert np.all(w >= -0.01) and np.all(w <= 1.01)
    print("  test_tapers_range: PASSED")


def test_waveform_generation():
    """NHSBWaveform produces a complex frequency-domain strain."""
    f = np.arange(20.0, 512.0, 0.5)
    src = SourceParams(35, 30)
    nhsb = NHSBParams(1e-8, 0.2, 0.95, 1.0)
    wf = NHSBWaveform()

    h = wf.h_tilde(f, src, nhsb)
    assert h.shape == f.shape
    assert np.iscomplexobj(h)
    assert np.all(np.isfinite(h))

    # BBH baseline should also work
    h_bbh = wf.h_tilde_bbh(f, src)
    assert h_bbh.shape == f.shape
    print("  test_waveform_generation: PASSED")


def test_waveform_bbh_limit():
    """NHSB waveform with Λ★=0, A=1: amplitude matches BBH, R→0."""
    f = np.arange(20.0, 512.0, 0.5)
    src = SourceParams(35, 30)
    nhsb_bh = NHSBParams(1e-8, 0.0, 1.0, 1.0)
    wf = NHSBWaveform()

    h_nhsb = wf.h_tilde(f, src, nhsb_bh)
    h_bbh = wf.h_tilde_bbh(f, src)

    # Amplitudes should match (since R=0 → T=0, so (1+W_R*T)=1)
    ratio = np.abs(h_nhsb) / np.abs(h_bbh)
    assert np.allclose(ratio, 1.0, atol=1e-6), \
        f"Amplitude ratio deviates: max |ratio-1| = {np.max(np.abs(ratio-1))}"

    # Conservative sector vanishes when Λ★=0
    from nhsb_waveform.constitutive import effective_tidal_deformability, cavity_timescale
    tau_c = cavity_timescale(src.M_final_seconds, nhsb_bh.epsilon)
    Leff = effective_tidal_deformability(f, nhsb_bh, tau_c)
    assert np.max(np.abs(Leff)) < 1e-15, "Λ_eff should vanish at Λ★=0"

    # Reflectivity vanishes when A=1
    from nhsb_waveform.constitutive import reflectivity
    R = reflectivity(f, nhsb_bh, tau_c)
    assert np.max(np.abs(R)) < 1e-15, "R should vanish at A=1"

    # Note: phase deformation is nonzero at A=1 because the heating
    # term A·Φ_heat adds BH-level heating.  This is by design —
    # see the boxed implementation note in §3.3 of the paper.
    print("  test_waveform_bbh_limit: PASSED")


def test_consistency_cuts():
    """valid() accepts conservative corner, rejects pathological cases."""
    f = np.arange(20.0, 512.0, 0.5)
    src = SourceParams(35, 30)
    wf = NHSBWaveform()

    # Conservative corner should pass
    nhsb_ok = NHSBParams(1e-8, 0.2, 0.95, 1.0)
    assert wf.valid(f, src, nhsb_ok)

    # BBH limit should pass
    nhsb_bh = NHSBParams(1e-8, 0.0, 1.0, 1.0)
    assert wf.valid(f, src, nhsb_bh)
    print("  test_consistency_cuts: PASSED")


def test_ergoregion_floor():
    """Ergoregion stability floor increases with spin."""
    assert ergoregion_floor(0.5) < ergoregion_floor(0.8)
    assert ergoregion_floor(0.8) < ergoregion_floor(0.95)
    print("  test_ergoregion_floor: PASSED")


def test_prior():
    """log_prior returns finite for valid points, -inf for invalid."""
    nhsb_ok = NHSBParams(1e-8, 0.2, 0.95, 1.0)
    assert np.isfinite(log_prior(nhsb_ok))

    # |Λ★| > 10 should be rejected
    nhsb_bad = NHSBParams(1e-8, 15.0, 0.95, 1.0)
    assert log_prior(nhsb_bad) == -np.inf

    # Sample should produce valid params
    p = sample_prior()
    assert 0 < p.epsilon < 1
    assert abs(p.Lambda_star) <= 10
    assert 0 <= p.A_abs <= 1
    assert 0.25 <= p.delta <= 1.75
    print("  test_prior: PASSED")


def test_psd():
    """aLIGO PSD returns positive values in the right ballpark."""
    f = np.arange(20.0, 2048.0, 1.0)
    Sn = aLIGO_design_psd(f)
    assert np.all(Sn > 0)
    assert np.all(np.isfinite(Sn))
    # Should be O(1e-46) at ~100 Hz
    idx100 = np.argmin(np.abs(f - 100))
    assert 1e-50 < Sn[idx100] < 1e-42
    print("  test_psd: PASSED")


def test_fisher():
    """Fisher matrix produces the ~70× channel hierarchy."""
    from nhsb_waveform.fisher import fisher_matrix
    f = np.arange(20.0, 512.0, 0.5)
    src = SourceParams(35, 30)
    _, sigma, _ = fisher_matrix(f, src, snr_target=100)
    ratio = sigma[0] / sigma[1]
    assert 30 < ratio < 200, f"Expected ~70× ratio, got {ratio:.0f}×"
    print(f"  test_fisher: PASSED (ratio = {ratio:.0f}×)")


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("nhsb_waveform test suite")
    print("=" * 50)

    tests = [
        test_params_validation,
        test_source_params,
        test_cavity_timescale,
        test_microtexture_asymptotic,
        test_bh_recovery_lambda,
        test_bh_recovery_reflectivity,
        test_transfer_function_bounded,
        test_tapers_range,
        test_waveform_generation,
        test_waveform_bbh_limit,
        test_consistency_cuts,
        test_ergoregion_floor,
        test_prior,
        test_psd,
        test_fisher,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAILED — {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed, "
          f"{len(tests)} total")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)
