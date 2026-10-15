"""Синтетические тесты спектрального модуля HRV.

Эталон: RR(t) = mean + A_lf*sin(2*pi*f_lf*t) + A_hf*sin(2*pi*f_hf*t).
Теоретическая мощность синусоиды амплитуды A равна A^2/2 (дисперсия),
поэтому при f_lf=0.10 Гц в полосе LF и f_hf=0.25 Гц в полосе HF ожидаем:
  LF ~ A_lf^2/2, HF ~ A_hf^2/2, LF/HF ~ (A_lf/A_hf)^2.
Проверяется конвейер modules.hrvanalysis.compute_frequency_domain_features
(resample 4 Гц + CubicSpline + poly-detrend + Welch).
"""
import numpy as np
import pytest

from modules.hrvanalysis import (
    compute_frequency_domain_features,
    compute_spectral_peak_features,
    _band_power,
    _build_cumulative_time,
)

MEAN_RR = 800.0
F_LF, F_HF = 0.10, 0.25
A_LF, A_HF = 50.0, 30.0
THEORY_LF = A_LF ** 2 / 2   # 1250 мс^2
THEORY_HF = A_HF ** 2 / 2   # 450 мс^2
THEORY_RATIO = THEORY_LF / THEORY_HF  # ~2.78


def synth_rr(mean=MEAN_RR, a_lf=0.0, f_lf=F_LF, a_hf=0.0, f_hf=F_HF,
             n=380, seed=0, noise_sd=0.0):
    """Физиологичная генерация: модуляция от текущего времени удара."""
    rng = np.random.default_rng(seed)
    rr = np.empty(n, dtype=float)
    t = 0.0
    for i in range(n):
        v = mean
        if a_lf:
            v += a_lf * np.sin(2 * np.pi * f_lf * t)
        if a_hf:
            v += a_hf * np.sin(2 * np.pi * f_hf * t)
        if noise_sd:
            v += noise_sd * rng.standard_normal()
        rr[i] = v
        t += v / 1000.0
    return rr


def test_pure_lf_power_and_peak():
    rr = synth_rr(a_lf=A_LF, n=380)
    f = compute_frequency_domain_features(rr)
    p = compute_spectral_peak_features(rr)
    assert f["lf_ms2"] == pytest.approx(THEORY_LF, rel=0.15), f["lf_ms2"]
    assert f["hf_ms2"] < 0.05 * f["lf_ms2"], (f["hf_ms2"], f["lf_ms2"])
    assert p["peak_lf_hz"] == pytest.approx(F_LF, abs=0.02), p["peak_lf_hz"]
    assert f["lf_norm"] > 80


def test_pure_hf_power_and_peak():
    rr = synth_rr(a_hf=A_HF, n=380)
    f = compute_frequency_domain_features(rr)
    p = compute_spectral_peak_features(rr)
    assert f["hf_ms2"] == pytest.approx(THEORY_HF, rel=0.15), f["hf_ms2"]
    assert f["lf_ms2"] < 0.05 * f["hf_ms2"], (f["lf_ms2"], f["hf_ms2"])
    assert p["peak_hf_hz"] == pytest.approx(F_HF, abs=0.02), p["peak_hf_hz"]
    assert f["hf_norm"] > 80


def test_mix_ratio_and_norms_sum():
    rr = synth_rr(a_lf=A_LF, a_hf=A_HF, n=380)
    f = compute_frequency_domain_features(rr)
    assert f["lf_ms2"] == pytest.approx(THEORY_LF, rel=0.15), f["lf_ms2"]
    assert f["hf_ms2"] == pytest.approx(THEORY_HF, rel=0.15), f["hf_ms2"]
    assert f["lf_hf_ratio"] == pytest.approx(THEORY_RATIO, rel=0.20), f["lf_hf_ratio"]
    assert f["lf_norm"] + f["hf_norm"] == pytest.approx(100.0, abs=0.5)
    p = compute_spectral_peak_features(rr)
    assert p["peak_lf_hz"] == pytest.approx(F_LF, abs=0.02)
    assert p["peak_hf_hz"] == pytest.approx(F_HF, abs=0.02)


def test_parseval_total_and_freq_grid():
    rr = synth_rr(a_lf=A_LF, a_hf=A_HF, n=380)
    f = compute_frequency_domain_features(rr)
    # total = VLF+LF+HF для записей >= 5 мин
    assert f["total_power_ms2"] == pytest.approx(
        f["vlf_ms2"] + f["lf_ms2"] + f["hf_ms2"], rel=0.02)
    # частотная сетка достаточна: >=5 бинов в LF, >=10 в HF
    assert f["n_lf_bins"] >= 5, f["n_lf_bins"]
    assert f["n_hf_bins"] >= 10, f["n_hf_bins"]
    assert f["freq_resolution_hz"] <= 0.01


def test_flatline_zero_power():
    rr = np.full(380, MEAN_RR)
    f = compute_frequency_domain_features(rr)
    assert f["total_power_ms2"] == pytest.approx(0.0, abs=1.0)
    assert f["lf_ms2"] == pytest.approx(0.0, abs=1.0)
    assert f["hf_ms2"] == pytest.approx(0.0, abs=1.0)


def test_short_record_vlf_nan():
    rr = synth_rr(a_lf=A_LF, a_hf=A_HF, n=100)  # ~80 c < 5 мин
    assert rr.sum() / 1000.0 < 300
    f = compute_frequency_domain_features(rr)
    assert np.isnan(f["vlf_ms2"])
    assert f["total_power_ms2"] == pytest.approx(f["lf_ms2"] + f["hf_ms2"], rel=0.02)


def test_band_power_exact_edges():
    freqs = np.array([0.0, 0.05, 0.10, 0.20, 0.30])
    psd = np.ones_like(freqs)  # плоская PSD=1 -> интеграл = ширина полосы
    assert _band_power(freqs, psd, 0.04, 0.15) == pytest.approx(0.11, rel=1e-9)
    assert _band_power(freqs, psd, 0.15, 0.40) == pytest.approx(0.25, rel=1e-9)


def test_timeline_cumsum_convention():
    rr = np.array([800.0, 850.0, 900.0])
    t = _build_cumulative_time(rr, None)
    assert np.allclose(t, np.cumsum(rr / 1000.0))
    # явный time_column проходит как есть (со сдвигом на t0)
    tc = np.array([100.0, 100.8, 101.65])
    assert np.allclose(_build_cumulative_time(rr, tc), tc - tc[0])


def test_time_column_consistency():
    rr = synth_rr(a_lf=A_LF, a_hf=A_HF, n=380)
    tc = np.cumsum(rr / 1000.0)
    a = compute_frequency_domain_features(rr)
    b = compute_frequency_domain_features(rr, time_column=tc)
    assert b["lf_ms2"] == pytest.approx(a["lf_ms2"], rel=0.05)
    assert b["hf_ms2"] == pytest.approx(a["hf_ms2"], rel=0.05)


def test_mix_with_noise_robustness():
    rr = synth_rr(a_lf=A_LF, a_hf=A_HF, n=380, noise_sd=5.0, seed=42)
    f = compute_frequency_domain_features(rr)
    assert f["lf_hf_ratio"] == pytest.approx(THEORY_RATIO, rel=0.30)
