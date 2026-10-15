#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Универсальный анализатор вариабельности сердечного ритма (ВСР)
Поддерживает RR-интервалы из CSV и TXT файлов.
Расчет спектральных, энтропийных, фрактальных показателей и индексов ИН, ПАПР, ВПР.
Интерполяция выбросов (без потери хронологии).
Генерирует HTML и XLSX отчеты с цветовой индикацией "светофор" (с защитой от ложных срабатываний).
Нормативы загружаются из внешнего конфигурационного файла hrv_norms.json.
"""

import os
import sys
import re
import io
import json
from openpyxl.styles import PatternFill, Font
from .html_report_builder import HTMLReportBuilder
import base64
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree as KDTree

__version__ = "1.1.1"

# Настройки matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
pd.options.mode.chained_assignment = None


# =============================================================================
# 1. МАТЕМАТИЧЕСКОЕ ЯДРО (Дескрипторы ВСР)
# =============================================================================

def compute_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Вычисляет временные показатели ВСР."""
    if len(rr_intervals) == 0:
        return {k: float('nan') for k in ["mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_percent", "sdann_ms"]}

    rr_diff = np.diff(rr_intervals)
    rr_diff_abs = np.abs(rr_diff)

    # ИСПРАВЛЕНИЕ 3: Добавлено ddof=1 для несмещенного стандартного отклонения
    sdnn = float(np.std(rr_intervals, ddof=1))
    rmssd = float(np.sqrt(np.mean(rr_diff**2))) if len(rr_diff) > 0 else float('nan')
    pnn50 = float(np.sum(rr_diff_abs > 50) / len(rr_diff) * 100) if len(rr_diff) > 0 else float('nan')
    mean_rr = float(np.mean(rr_intervals))

    duration_sec = np.sum(rr_intervals) / 1000.0
    # SDANN вычисляется только для записей >= 5 минут.
    # Приближение: разбиение по числу комплексов через средний RR, неполный
    # последний сегмент отбрасывается. При наличии меток времени точнее
    # сегментировать по реальному времени.
    if duration_sec >= 300 and mean_rr > 0:
        n_segments = int(duration_sec // 300)
        segment_means = []
        for i in range(n_segments):
            start_idx = int(i * 300 * 1000 / mean_rr)
            end_idx = int((i + 1) * 300 * 1000 / mean_rr)
            if end_idx <= len(rr_intervals):
                segment_means.append(np.mean(rr_intervals[start_idx:end_idx]))
        # ИСПРАВЛЕНИЕ 3: Добавлено ddof=1 для SDANN
        sdann = float(np.std(segment_means, ddof=1)) if segment_means else float('nan')
    else:
        sdann = float('nan')

    return {
        "mean_rr_ms": round(mean_rr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50_percent": round(pnn50, 2),
        "sdann_ms": round(sdann, 2) if not pd.isna(sdann) else float('nan')
    }


def _build_cumulative_time(rr_intervals: np.ndarray,
                             time_column: Optional[np.ndarray] = None) -> np.ndarray:
    """Временная шкала тахограммы в секундах.

    Соглашение: каждый RR относится к моменту окончания интервала,
    t[k] = sum(rr[0..k]) / 1000. Используется cumsum, а НЕ
    cumulative_trapezoid (усреднение соседних RR искажает шкалу).
    Внешний time_column валидируется (конечность, строгая монотонность);
    единицы автоопределяются: медианный шаг > 5 c считается миллисекундами.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    if time_column is not None:
        try:
            tc = np.asarray(time_column, dtype=float)
            if (len(tc) == len(rr) and np.all(np.isfinite(tc))
                    and np.all(np.isfinite(rr)) and np.all(rr > 0)
                    and np.all(np.diff(tc) > 0)):
                t = tc - tc[0]
                if len(t) > 1 and float(np.median(np.diff(t))) > 5.0:
                    t = t / 1000.0  # столбец был в мс
                return t
        except Exception:
            pass
    # fallback: накопительное суммирование (время окончания интервалов)
    return np.cumsum(rr / 1000.0)


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Интеграл PSD на [lo, hi) с интерполяцией значений на точных границах."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if len(freqs) < 2 or not np.all(np.isfinite(psd)):
        return 0.0
    inside = (freqs > lo) & (freqs < hi)
    x = [lo]
    y = [float(np.interp(lo, freqs, psd))]
    x.extend(freqs[inside].tolist())
    y.extend(psd[inside].tolist())
    x.append(hi)
    y.append(float(np.interp(hi, freqs, psd)))
    return float(trapezoid(np.asarray(y), x=np.asarray(x)))


def compute_frequency_domain_features(rr_intervals: np.ndarray, fs: float = 4.0,
                                       time_column: Optional[np.ndarray] = None,
                                       detrend_order: int = 3) -> Dict[str, float]:
    """Вычисляет спектральные показатели ВСР методом Уэлча.

    detrend_order: степень полиномиального тренда (по умолчанию 3).
    Внимание: удаление тренда влияет на VLF/нижнюю часть LF.
    PSD имеет единицы мс^2/Гц, интегралы — мс^2.
    """

    rr_intervals = np.asarray(rr_intervals, dtype=float)
    if len(rr_intervals) < 2 or not np.all(np.isfinite(rr_intervals)) or np.any(rr_intervals <= 0):
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}

    cumulative_time = _build_cumulative_time(rr_intervals, time_column)

    if len(cumulative_time) != len(rr_intervals) or not np.all(np.isfinite(cumulative_time)):
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}

    if cumulative_time[-1] <= 0 or not np.all(np.diff(cumulative_time) > 0):
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}

    # Равномерная сетка от первого до последнего отсчёта (без экстраполяции назад к 0)
    t_uniform = np.arange(cumulative_time[0], cumulative_time[-1], 1.0 / fs)
    if len(t_uniform) < 16:
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}

    # Кубическая сплайн-интерполяция (сохраняет высокочастотные HF-волны)
    try:
        cs = CubicSpline(cumulative_time, rr_intervals)
        rr_uniform = cs(t_uniform)
    except Exception:
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}
    if not np.all(np.isfinite(rr_uniform)):
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}

    # Удаление полиномиального тренда (влияет на VLF/LF — см. docstring)
    x_uniform = np.arange(len(rr_uniform))
    coeffs = np.polyfit(x_uniform, rr_uniform, detrend_order)
    trend = np.polyval(coeffs, x_uniform)
    rr_detrended = rr_uniform - trend

    # Адаптивный nperseg: не фиксируем 256 для длинных записей, cap 512
    if len(rr_detrended) < 16:  # Защита от слишком коротких записей
        return {k: float('nan') for k in ["vlf_ms2", "lf_ms2", "hf_ms2", "total_power_ms2", "lf_hf_ratio", "lf_norm", "hf_norm", "freq_resolution_hz", "n_lf_bins", "n_hf_bins", "freqs", "psd"]}
    nperseg = min(512, len(rr_detrended))

    # Метод Уэлча по умолчанию применяет окно Хеннинга, что правильно
    freqs, psd = signal.welch(rr_detrended, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling='density')
    freq_res = float(fs / nperseg)
    n_lf = int(np.sum((freqs >= 0.04) & (freqs < 0.15)))
    n_hf = int(np.sum((freqs >= 0.15) & (freqs < 0.4)))

    lf = _band_power(freqs, psd, 0.04, 0.15)
    hf = _band_power(freqs, psd, 0.15, 0.4)

    # VLF валиден только для записей >= 5 минут
    duration_sec = np.sum(rr_intervals) / 1000.0
    if duration_sec >= 300:
        vlf = _band_power(freqs, psd, 0.003, 0.04)
    else:
        vlf = float('nan')

    # Пересчет Total Power с учетом валидности VLF
    if pd.isna(vlf):
        total_power = lf + hf
    else:
        total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf if hf > 0 else float('nan')

    denominator = total_power - vlf if not pd.isna(vlf) else total_power
    lf_norm = lf / denominator * 100 if denominator > 0 else float('nan')
    hf_norm = hf / denominator * 100 if denominator > 0 else float('nan')

    # lf_norm/hf_norm — относительные доли от (LF+HF), в сумме ~100%; не являются
    # независимыми мерами активности отделов ВНС.
    return {
        "vlf_ms2": round(vlf, 2) if not pd.isna(vlf) else float('nan'),
        "lf_ms2": round(lf, 2),
        "hf_ms2": round(hf, 2),
        "total_power_ms2": round(total_power, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 3) if not pd.isna(lf_hf_ratio) else float('nan'),
        "lf_norm": round(lf_norm, 2) if not pd.isna(lf_norm) else float('nan'),
        "hf_norm": round(hf_norm, 2) if not pd.isna(hf_norm) else float('nan'),
        "freq_resolution_hz": round(freq_res, 5),
        "n_lf_bins": n_lf,
        "n_hf_bins": n_hf,
        "freqs": freqs,
        "psd": psd
    }


def compute_spectral_peak_features(rr_intervals: np.ndarray, fs: float = 4.0,
                                    time_column: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Пиковые частоты спектральных полос и нормированные доли (nu)."""
    freq_features = compute_frequency_domain_features(rr_intervals, fs=fs, time_column=time_column)
    freqs = freq_features["freqs"]
    psd = freq_features["psd"]
    if not isinstance(freqs, np.ndarray) or not isinstance(psd, np.ndarray):
        return {"peak_vlf_hz": float('nan'), "peak_lf_hz": float('nan'),
                "peak_hf_hz": float('nan'), "fraction_lf_nu": float('nan'),
                "fraction_hf_nu": float('nan')}

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    def _peak(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(freqs[mask][np.argmax(psd[mask])])

    return {
        "peak_vlf_hz": round(_peak(vlf_mask), 4),
        "peak_lf_hz": round(_peak(lf_mask), 4),
        "peak_hf_hz": round(_peak(hf_mask), 4),
        "fraction_lf_nu": freq_features["lf_norm"],
        "fraction_hf_nu": freq_features["hf_norm"],
    }


def compute_sample_entropy(rr_intervals: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Вычисляет Sample Entropy (SampEn)."""
    try:
        import antropy as _ant
        val = _ant.sample_entropy(np.asarray(rr_intervals, dtype=float), order=m, r=r)
        return float(val) if np.isfinite(val) else float('nan')
    except Exception:
        pass

    rr = np.asarray(rr_intervals, dtype=float)
    n = len(rr)
    if n < m + 2:
        return float('nan')

    threshold = float(r) * float(np.std(rr, ddof=1)) # ddof=1 для consistency
    if threshold <= 0:
        return float('nan')

    def count_matches(template_length: int) -> float:
        pats = np.array([rr[i:i + template_length] for i in range(n - template_length + 1)])
        tree = KDTree(pats)
        neighbours = tree.query_ball_point(pats, threshold, p=np.inf)
        return float(sum(len(x) - 1 for x in neighbours))

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0 or A == 0:
        return float('nan')
    return float(-np.log(A / B))


def _dfa_entropy(rr: np.ndarray, scales_short=(4, 16), scales_long=(17, 64)) -> Tuple[float, float]:
    """DFA 1 (короткие шкалы) и 2 (длинные шкалы)."""
    try:
        y = np.cumsum(rr - np.mean(rr))
        n = len(y)

        def calc_alpha(scales):
            fluct = []
            valid_scales = []
            for scale in scales:
                n_seg = n // scale
                if n_seg < 1:
                    continue
                rms = 0.0
                for v in range(n_seg):
                    idx = np.arange(v * scale, (v + 1) * scale)
                    x = np.arange(len(idx))
                    coeffs = np.polyfit(x, y[idx], 1)
                    trend = np.polyval(coeffs, x)
                    rms += np.sum((y[idx] - trend) ** 2)
                fluct.append(np.sqrt(rms / (n_seg * scale)))
                valid_scales.append(scale)
            return np.polyfit(np.log(valid_scales), np.log(fluct), 1)[0] if len(fluct) >= 2 else np.nan

        return (calc_alpha(np.arange(scales_short[0], scales_short[1] + 1, dtype=int)),
                calc_alpha(np.arange(scales_long[0], scales_long[1] + 1, dtype=int)))
    except Exception:
        return np.nan, np.nan


def _approx_entropy(u: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate Entropy (ApEn)."""
    if len(u) < m + 2:
        return float('nan')
    N = len(u)
    r = float(r) * np.std(u, ddof=1)
    if r <= 0: return float('nan')

    def _phi(mm):
        x = np.array([u[i:i + mm] for i in range(N - mm + 1)])
        C = np.mean([np.sum(np.max(np.abs(x - xi), axis=1) <= r) for xi in x]) / (N - mm + 1)
        return np.mean(np.log(np.maximum(C, 1e-9)))

    val = _phi(m) - _phi(m + 1)
    return float(val) if np.isfinite(val) else float('nan')


def _fuzzy_entropy(u: np.ndarray, m: int = 2, r: float = 0.15, n: float = 2.0) -> float:
    """Fuzzy Entropy (FuzzyEn)."""
    if len(u) < m + 2:
        return float('nan')
    N = len(u)
    r = float(r) * np.std(u, ddof=1)
    if r <= 0: return float('nan')

    def _phi(mm):
        x = np.array([u[i:i + mm] for i in range(N - mm + 1)])
        dist = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
        np.fill_diagonal(dist, np.inf)
        mu = np.exp(-(dist ** n) / (r + 1e-9))
        return np.sum(mu) / ((N - mm + 1) * (N - mm))

    B = _phi(m)
    if B <= 0: return float('nan')
    A = _phi(m + 1)
    val = -np.log(A / (B + 1e-9))
    return float(val) if np.isfinite(val) else float('nan')


def _multiscale_entropy(rr: np.ndarray, max_scale: int = 20, m: int = 2, r: float = 0.15) -> Dict[str, float]:
    """Мультимасштабная энтропия (MSE)."""
    entropies = []
    for scale in range(1, max_scale + 1):
        cg_rr = np.array([np.mean(rr[i:i + scale]) for i in range(0, len(rr) - scale + 1, scale)])
        if len(cg_rr) < m + 2:
            entropies.append(float('nan'))
            continue
        entropies.append(compute_sample_entropy(cg_rr, m=m, r=r))

    entropies = np.array(entropies, dtype=float)
    scales = np.arange(1, max_scale + 1)
    mask = ~np.isnan(entropies)
    if int(np.sum(mask)) < 2:
        return {"mse_mean": float('nan'), "mse_slope": float('nan')}
    return {
        "mse_mean": float(np.mean(entropies[mask])),
        # наклон по исходным номерам масштабов, а не по порядковым номерам валидных
        "mse_slope": float(np.polyfit(scales[mask], entropies[mask], 1)[0])
    }


def compute_entropy_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Расширенный энтропийный/фрактальный анализ с учётом условий применимости.

    Пороги интерпретации SampEn/ApEn/FuzzyEn/MSE нельзя переносить между
    реализациями без проверки: нормировки и параметры r/m различаются.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    n = len(rr)
    feats: Dict[str, float] = {}

    try:
        val = compute_sample_entropy(rr, m=2, r=0.2)
        feats["sample_entropy"] = round(val, 3) if not pd.isna(val) else float('nan')
    except Exception:
        feats["sample_entropy"] = float('nan')

    if n >= 64:
        a1, a2 = _dfa_entropy(rr)
        feats["dfa_alpha1"] = round(float(a1), 3) if np.isfinite(a1) else float('nan')
        feats["dfa_alpha2"] = round(float(a2), 3) if np.isfinite(a2) else float('nan')
    else:
        feats["dfa_alpha1"] = float('nan')
        feats["dfa_alpha2"] = float('nan')

    if n >= 150:
        rr_norm = (rr - np.mean(rr)) / (np.std(rr, ddof=1) + 1e-10)
        try:
            val = _approx_entropy(rr_norm, m=2, r=0.2)
            feats["approx_entropy"] = round(val, 3) if not pd.isna(val) else float('nan')
        except Exception:
            feats["approx_entropy"] = float('nan')
        try:
            val = _fuzzy_entropy(rr_norm, m=2, r=0.15, n=2.0)
            feats["fuzzy_entropy"] = round(val, 3) if not pd.isna(val) else float('nan')
        except Exception:
            feats["fuzzy_entropy"] = float('nan')
    else:
        feats["approx_entropy"] = float('nan')
        feats["fuzzy_entropy"] = float('nan')

    if n >= 500:
        try:
            mse = _multiscale_entropy(rr, max_scale=20, m=2, r=0.15)
            feats["mse_mean"] = round(mse["mse_mean"], 3) if not pd.isna(mse["mse_mean"]) else float('nan')
            feats["mse_slope"] = round(mse["mse_slope"], 3) if not pd.isna(mse["mse_slope"]) else float('nan')
        except Exception:
            feats["mse_mean"] = float('nan')
            feats["mse_slope"] = float('nan')
    else:
        feats["mse_mean"] = float('nan')
        feats["mse_slope"] = float('nan')

    return feats


def compute_baevsky_indices(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Вычисляет классические индексы Баевского: ИН, ПАПР, ВПР.

    Условность: ширина бина фиксирована 50 мс, сетка привязана к
    floor(min/50)*50; Мо/АМо (и ИН) зависят от положения границ бинов.
    """
    # Стандарт вариационной пульсометрии требует не менее 100 интервалов
    if len(rr_intervals) < 100:
        return {"in_index": float('nan'), "papr": float('nan'), "vpr": float('nan'),
                "mo_ms": float('nan'), "amo_percent": float('nan'), "dx_ms": float('nan')}

    x_max = float(np.max(rr_intervals))
    x_min = float(np.min(rr_intervals))
    range_rr = x_max - x_min

    if range_rr == 0:
        return {"in_index": float('nan'), "papr": float('nan'), "vpr": float('nan'),
                "mo_ms": x_max, "amo_percent": 100.0, "dx_ms": 0.0}

    bin_width = 50.0
    min_bin_edge = np.floor(x_min / bin_width) * bin_width
    max_bin_edge = np.ceil(x_max / bin_width) * bin_width
    bins = np.arange(min_bin_edge, max_bin_edge + bin_width, bin_width)

    counts, bin_edges = np.histogram(rr_intervals, bins=bins)
    mode_idx = np.argmax(counts)
    mo = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
    amo_percent = (counts[mode_idx] / len(rr_intervals)) * 100.0

    mo_sec = mo / 1000.0
    range_sec = range_rr / 1000.0

    if mo_sec > 0 and range_sec > 0:
        in_index = amo_percent / (2 * mo_sec * range_sec)
    else:
        in_index = float('nan')

    papr = amo_percent / mo if mo > 0 else float('nan')
    vpr = range_rr / mo if mo > 0 else float('nan')

    return {
        "in_index": round(in_index, 4) if not pd.isna(in_index) else float('nan'),
        "papr": round(papr, 4) if not pd.isna(papr) else float('nan'),
        "vpr": round(vpr, 4) if not pd.isna(vpr) else float('nan'),
        "mo_ms": round(mo, 2),
        "amo_percent": round(amo_percent, 2),
        "dx_ms": round(range_rr, 2)
    }

def compute_all_hrv_features(rr_intervals: np.ndarray,
                              time_column: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Вычисляет все дескрипторы ВСР."""
    features = {}
    features.update(compute_time_domain_features(rr_intervals))

    freq_features = compute_frequency_domain_features(rr_intervals, time_column=time_column)
    features.update({k: v for k, v in freq_features.items() if k not in ['freqs', 'psd']})

    features.update(compute_entropy_features(rr_intervals))
    features.update(compute_baevsky_indices(rr_intervals))

    # SD1 и SD2 (Пуанкаре). SD1 = SDSD/sqrt(2); равен RMSSD/sqrt(2) только при
    # нулевом среднем последовательных разностей и том же ddof.
    if len(rr_intervals) > 2:
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        # ИСПРАВЛЕНИЕ 3: Добавлено ddof=1 для SD1 и SD2
        features["sd1_ms"] = round(float(np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)), 2)
        features["sd2_ms"] = round(float(np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)), 2)
    else:
        features["sd1_ms"] = float('nan')
        features["sd2_ms"] = float('nan')

    features["n_rr_intervals"] = len(rr_intervals)
    features["duration_min"] = round(np.sum(rr_intervals) / 60000.0, 2)

    return features


# =============================================================================
# 2. ФИЛЬТРАЦИЯ ВЫБРОСОВ (ИНТЕРПОЛЯЦИЯ)
# =============================================================================

def remove_outliers_moving_average(rr_intervals: np.ndarray, window_size: int = 5,
                                    threshold: float = 3.0,
                                    time_column: Optional[np.ndarray] = None
                                    ) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    """
    Удаляет выбросы на основе скользящей медианы (устойчиво к экстрасистолам)
    и заменяет их линейной интерполяцией (без изменения длины ряда).
    Возвращает (исправленный ряд, число заменённых отсчётов, time_column).
    Ограничение: сигма-критерий от отклонений может пропускать выбросы при
    раздутом std или отмечать лишнее при нетипичном распределении.
    """
    if len(rr_intervals) < window_size:
        return rr_intervals, 0, time_column

    # ИСПРАВЛЕНИЕ 4: Использование скользящей медианы вместо среднего
    # medfilt требует нечетный размер окна
    win_size = window_size if window_size % 2 != 0 else window_size + 1
    moving_med = signal.medfilt(np.asarray(rr_intervals, dtype=float), kernel_size=win_size)

    deviations = np.abs(np.asarray(rr_intervals, dtype=float) - moving_med)
    std_dev = np.std(deviations, ddof=1) if len(deviations) > 1 else 0.0

    if std_dev == 0 or not np.isfinite(std_dev):
        # fallback на MAD, чтобы константный ряд с одиночным выбросом чистился
        med = float(np.median(deviations))
        mad = float(np.median(np.abs(deviations - med)))
        scale = 1.4826 * mad
        if scale == 0 or not np.isfinite(scale):
            return rr_intervals, 0, time_column
        std_dev = scale

    valid_mask = deviations < (threshold * std_dev)
    n_removed = len(rr_intervals) - np.sum(valid_mask)

    # Интерполяция выбросов вместо их удаления!
    rr_corrected = rr_intervals.copy()
    if n_removed > 0 and np.sum(valid_mask) > 2:
        invalid_indices = np.where(~valid_mask)[0]
        valid_indices = np.where(valid_mask)[0]
        rr_corrected[invalid_indices] = np.interp(invalid_indices, valid_indices, rr_intervals[valid_indices])

    return rr_corrected, n_removed, time_column


# =============================================================================
# 3. ЗАГРУЗКА НОРМАТИВОВ ИЗ КОНФИГА
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "hrv_norms.json"

DEFAULT_NORMS: Dict[str, Dict[str, Any]] = {
    "mean_rr_ms": {
        "desc": "Средний RR-интервал", "unit": "мс",
        "green": [600, 1200], "yellow": [500, 1400], "category": "rhythm",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренная тахикардия", "severe": "Значительно ниже нормы. Выраженная тахикардия"},
            "high": {"moderate": "Выше нормы. Умеренная брадикардия", "severe": "Значительно выше нормы. Выраженная брадикардия"}
        }
    },
    "sdnn_ms": {
        "desc": "SDNN (общая вариабельность)", "unit": "мс",
        "green": [50, 200], "yellow": [30, 50], "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение общей ВСР", "severe": "Значительно ниже нормы. Выраженное снижение ВСР"},
            "high": {"moderate": "Выше нормы. Умеренное повышение ВСР", "severe": "Значительно выше нормы. Выраженное повышение ВСР"}
        }
    },
    "rmssd_ms": {
        "desc": "RMSSD (парасимпатическая активность)", "unit": "мс",
        "green": [20, 100], "yellow": [10, 20], "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение вагусного влияния", "severe": "Значительно ниже нормы. Выраженная симпатикотония"},
            "high": {"moderate": "Выше нормы. Умеренная ваготония", "severe": "Значительно выше нормы. Выраженная ваготония"}
        }
    },
    "pnn50_percent": {
        "desc": "pNN50 (парасимпатический индекс)", "unit": "%",
        "green": [3, 20], "yellow": [1, 3], "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренная симпатикотония", "severe": "Значительно ниже нормы. Выраженная симпатикотония"},
            "high": {"moderate": "Выше нормы. Умеренная ваготония", "severe": "Значительно выше нормы. Выраженная ваготония"}
        }
    },
    "lf_hf_ratio": {
        "desc": "LF/HF (грубый индекс, не мера баланса)", "unit": "",
        "green": [1.0, 2.0], "yellow": [0.5, 3.0], "category": "balance",
        "interpretation": {
            "normal": "Норма. Отношение в заданных границах; само по себе не доказывает вегетативный баланс",
            "low": {"moderate": "Ниже нормы. Относительное преобладание HF; трактовать только в контексте записи", "severe": "Значительно ниже нормы. Выраженное относительное преобладание HF; не диагноз ваготонии"},
            "high": {"moderate": "Выше нормы. Относительное преобладание LF; трактовать только в контексте записи", "severe": "Значительно выше нормы. Выраженное относительное преобладание LF; не диагноз симпатикотонии"}
        }
    },
    "hf_ms2": {
        "desc": "HF мощность (преимущественно вагусная)", "unit": "мс²",
        "green": [500, 3000], "yellow": [200, 500], "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение мощности HF", "severe": "Значительно ниже нормы. Выраженное снижение мощности HF"},
            "high": {"moderate": "Выше нормы. Умеренно повышенная мощность HF", "severe": "Значительно выше нормы. Выраженно повышенная мощность HF"}
        }
    },
    "lf_ms2": {
        "desc": "LF мощность (смешанная, не чисто симпатическая)", "unit": "мс²",
        "green": [300, 2000], "yellow": [100, 300], "category": "mixed",
        "interpretation": {
            "normal": "Норма. Мощность LF в заданных границах; LF отражает смешанные (барорефлекторные, вагусные и симпатические) механизмы",
            "low": {"moderate": "Ниже нормы. Умеренно сниженная мощность LF (неспецифично)", "severe": "Значительно ниже нормы. Выраженно сниженная мощность LF (неспецифично)"},
            "high": {"moderate": "Выше нормы. Умеренно повышенная мощность LF; не является доказательством симпатикотонии", "severe": "Значительно выше нормы. Выраженно повышенная мощность LF; не является доказательством симпатикотонии"}
        }
    },
    "in_index": {
        "desc": "ИН Баевского (индекс напряжения)", "unit": "",
        "green": [50, 150], "yellow": [25, 300], "category": "stress",
        "interpretation": {
            "normal": "Норма. Удовлетворительная адаптация",
            "low": {"moderate": "Ниже нормы. Умеренное снижение напряжения РС", "severe": "Значительно ниже нормы. Истощение регуляторных систем"},
            "high": {"moderate": "Выше нормы. Умеренное напряжение регуляторных систем", "severe": "Значительно выше нормы. Выраженное напряжение РС"}
        }
    },
    "papr": {
        "desc": "ПАПР (адекватность регуляции)", "unit": "",
        "green": [1.0, 3.0], "yellow": [0.5, 1.0], "category": "regulation",
        "interpretation": {
            "normal": "Норма. Адекватная регуляция",
            "low": {"moderate": "Ниже нормы. Умеренное снижение адекватности регуляции", "severe": "Значительно ниже нормы. Выраженная дезадаптация"},
            "high": {"moderate": "Выше нормы. Умеренное перенапряжение регуляции", "severe": "Значительно выше нормы. Выраженная дезадаптация"}
        }
    },
    "vpr": {
        "desc": "ВПР (вегетативный индекс реактивности)", "unit": "",
        "green": [1.0, 3.0], "yellow": [0.5, 1.0], "category": "reactivity",
        "interpretation": {
            "normal": "Норма. Адекватная реактивность",
            "low": {"moderate": "Ниже нормы. Умеренное снижение реактивности", "severe": "Значительно ниже нормы. Выраженное снижение реактивности"},
            "high": {"moderate": "Выше нормы. Умеренное повышение реактивности", "severe": "Значительно выше нормы. Гиперреактивность"}
        }
    },
    "sample_entropy": {
        "desc": "Sample Entropy (сложность ритма)", "unit": "",
        "green": [1.0, 2.5], "yellow": [0.5, 1.0], "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма", "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма", "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "dfa_alpha1": {
        "desc": "DFA 1 (фрактальная корреляция)", "unit": "",
        "green": [1.0, 1.5], "yellow": [0.75, 1.0], "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая фрактальная структура",
            "low": {"moderate": "Ниже нормы. Умеренная случайность ритма", "severe": "Значительно ниже нормы. Выраженная хаотизация"},
            "high": {"moderate": "Выше нормы. Умеренная избыточная регулярность", "severe": "Значительно выше нормы. Избыточная предсказуемость"}
        }
    },
    "sd1_ms": {
        "desc": "SD1 (краткосрочная вариабельность)", "unit": "мс",
        "green": [10, 50], "yellow": [5, 10], "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение краткосрочной вариабельности", "severe": "Значительно ниже нормы. Выраженное снижение"},
            "high": {"moderate": "Выше нормы. Умеренное повышение краткосрочной вариабельности", "severe": "Значительно выше нормы. Выраженное повышение"}
        }
    },
    "sd2_ms": {
        "desc": "SD2 (долгосрочная вариабельность)", "unit": "мс",
        "green": [30, 150], "yellow": [20, 30], "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение долгосрочной вариабельности", "severe": "Значительно ниже нормы. Выраженное снижение"},
            "high": {"moderate": "Выше нормы. Умеренное повышение долгосрочной вариабельности", "severe": "Значительно выше нормы. Выраженное повышение"}
        }
    },
    "dfa_alpha2": {
        "desc": "DFA 2 (фрактальная корреляция, длинные шкалы)", "unit": "",
        "green": [0.8, 1.2], "yellow": [0.6, 0.8], "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая фрактальная структура на длинных шкалах",
            "low": {"moderate": "Ниже нормы. Умеренная случайность на длинных шкалах", "severe": "Значительно ниже нормы. Выраженная хаотизация"},
            "high": {"moderate": "Выше нормы. Умеренная избыточная регулярность", "severe": "Значительно выше нормы. Избыточная предсказуемость"}
        }
    },
    "approx_entropy": {
        "desc": "Approximate Entropy (регулярность ритма)", "unit": "",
        "green": [0.8, 1.8], "yellow": [0.4, 0.8], "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма", "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма", "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "fuzzy_entropy": {
        "desc": "Fuzzy Entropy (мягкая нерегулярность ритма)", "unit": "",
        "green": [0.5, 1.5], "yellow": [0.2, 0.5], "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма", "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма", "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "mse_mean": {
        "desc": "MSE mean (средняя мультимасштабная энтропия)", "unit": "",
        "green": [0.8, 1.8], "yellow": [0.4, 0.8], "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая регуляторная сложность по масштабам",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение на мультимасштабе", "severe": "Значительно ниже нормы. Выраженная ригидность"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение", "severe": "Значительно выше нормы. Хаотизация"}
        }
    },
    "mse_slope": {
        "desc": "MSE slope (наклон мультимасштабной энтропии)", "unit": "",
        "green": [-0.05, 0.05], "yellow": [-0.1, -0.05], "category": "complexity",
        "interpretation": {
            "normal": "Норма. Близкий к нулю наклон (здоровая адаптация)",
            "low": {"moderate": "Отрицательный наклон. Снижение сложности с масштабом", "severe": "Выраженный отрицательный наклон (признак патологии)"},
            "high": {"moderate": "Положительный наклон. Рост сложности с масштабом", "severe": "Выраженный положительный наклон"}
        }
    }
}


def load_norms_config(config_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Загружает нормативы из JSON-конфига."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f" Нормативы загружены из: {config_path.name}")
            return data
        except Exception as e:
            print(f" Ошибка чтения конфига {config_path}: {e}. Используются встроенные нормативы.")
            return DEFAULT_NORMS
    else:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_NORMS, f, ensure_ascii=False, indent=2)
            print(f" Создан файл нормативов: {config_path.name} (отредактируйте при необходимости)")
        except Exception as e:
            print(f" Не удалось создать файл конфига: {e}")
        return DEFAULT_NORMS


# =============================================================================
# 4. ИНТЕРПРЕТАЦИЯ ПОКАЗАТЕЛЕЙ (СИСТЕМА "СВЕТОФОР" С ЗАЩИТОЙ ОТ NaN)
# =============================================================================

def get_indicator_interpretation(value: float, metric_name: str, norms: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Определяет цвет, статус и текстовую интерпретацию для показателя ВСР."""
    if metric_name not in norms:
        return {"color": "#6c757d", "status": "—", "interpretation": "Показатель не имеет норматива", "direction": ""}

    # Защита от ложных заключений: если данных недостаточно, возвращаем серый статус
    if pd.isna(value) or value is None:
        return {"color": "#6c757d", "status": "Нет данных", "interpretation": "Недостаточно данных для расчета", "direction": ""}

    norm = norms[metric_name]
    g_min, g_max = norm["green"]

    # Определяем границы желтой зоны с учетом возможной асимметрии
    y_min = min(norm["yellow"][0], g_min)
    y_max = max(norm["yellow"][1], g_max)

    interp = norm.get("interpretation", {})

    if g_min <= value <= g_max:
        return {"color": "#198754", "status": "Норма", "interpretation": interp.get("normal", "Норма"), "direction": "в пределах нормы"}

    if value < g_min:
        direction = "ниже нормы"
        severity_key = "low"
    else:
        direction = "выше нормы"
        severity_key = "high"

    if y_min <= value <= y_max:
        severity = "moderate"
        severity_label = "умеренное"
    else:
        severity = "severe"
        severity_label = "выраженное"

    interp_block = interp.get(severity_key, {})
    text = interp_block.get(severity, f"{severity_label.capitalize()} отклонение ({direction})")

    return {
        "color": "#ffc107" if severity == "moderate" else "#dc3545",
        "status": "Умеренное отклонение" if severity == "moderate" else "Выраженное отклонение",
        "interpretation": text,
        "direction": direction
    }


# =============================================================================
# 5. ПАРСИНГ И ПОДГОТОВКА ДАННЫХ
# =============================================================================

def detect_rr_column(df: pd.DataFrame) -> str:
    """Автоопределение столбца с RR-интервалами.

    Строгий порядок: точное совпадение > токен с границами слов.
    Голый 'ms' как подстрока НЕ используется, чтобы не выбрать 'timestamp_ms'.
    """
    import re as _re
    cols_lower = {c: str(c).lower().strip() for c in df.columns}
    # 1) точные имена
    exact = ['rr', 'nn', 'rri', 'ibi', 'rr_interval', 'nn_interval']
    for c, l in cols_lower.items():
        if l in exact:
            return c
    # 2) токены с границами слов (без голого 'ms' и без одиночных букв)
    tokens = ['rr', 'nn', 'rri', 'ibi', 'interbeat', 'interval', 'rr_interval', 'nn_interval']
    for c, l in cols_lower.items():
        norm = _re.sub(r'[^a-z0-9]+', ' ', l)
        words = set(norm.split())
        if words & set(tokens):
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    raise ValueError("Не найден столбец с RR-интервалами")


def detect_time_column(df: pd.DataFrame, exclude_col: str = None) -> Optional[str]:
    """Автоопределение столбца времени (без одиночной 't' как подстроки)."""
    import re as _re
    cols_lower = {c: str(c).lower().strip() for c in df.columns}
    exact_priority = ['peak_time', 'timestamp', 'time', 'elapsed', 'clock', 'datetime',
                      'epoch', 'timesec', 'time_s', 'time_sec', 'time_ms']
    for pattern in exact_priority:
        for c, l in cols_lower.items():
            if c == exclude_col:
                continue
            if l == pattern or l.startswith(pattern + '_') or l.endswith('_' + pattern):
                return c
    # fallback только по границам слов, исключая rr/nn/ibi-столбцы
    for c, l in cols_lower.items():
        if c == exclude_col:
            continue
        norm = _re.sub(r'[^a-z0-9]+', ' ', l)
        words = set(norm.split())
        if words & {'time', 'timestamp', 'elapsed', 'clock', 'datetime', 'epoch'}:
            if not (words & {'rr', 'nn', 'ibi', 'interval'}):
                return c
    return None


def detect_rr_units(rr_values: np.ndarray, col_name: str = "") -> str:
    """Автоопределение единиц измерения RR-интервалов: 'ms' или 'sec'."""
    col_lower = col_name.lower().strip()
    sec_hints = ['_s', '_sec', '_seconds', 'sec', 'seconds', '_sec_', '_s_']
    ms_hints = ['_ms', '_millis', 'millis', 'ms']

    for hint in sec_hints:
        if hint in col_lower: return 'sec'
    for hint in ms_hints:
        if hint in col_lower: return 'ms'

    med = float(np.median(rr_values))
    if med < 3.0: return 'sec'
    return 'ms'


def load_rr_intervals_from_file(filepath: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Загрузка RR-интервалов из CSV или TXT файла."""
    filepath = Path(filepath)
    for delimiter in [',', ';', '\t', ' ', None]:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter)
            if not df.empty: break
        except: continue
    else:
        raise ValueError(f"Не удалось прочитать файл {filepath.name}")

    if df.empty:
        raise ValueError(f"Файл {filepath.name} пустой")

    rr_col = detect_rr_column(df)
    time_col = detect_time_column(df, exclude_col=rr_col)

    # Совместная очистка NaN, чтобы RR и время оставались построчно согласованы
    cols = [rr_col] + ([time_col] if time_col is not None else [])
    sub = df[cols].copy()
    sub[rr_col] = pd.to_numeric(sub[rr_col], errors='coerce')
    if time_col is not None:
        sub[time_col] = pd.to_numeric(sub[time_col], errors='coerce')
    sub = sub.dropna()
    if sub.empty:
        raise ValueError(f"Столбец {rr_col} не содержит числовых значений")

    rr_raw = sub[rr_col].to_numpy(dtype=float)
    units = detect_rr_units(rr_raw, col_name=str(rr_col))
    if units == 'sec':
        rr_intervals = rr_raw * 1000.0
    else:
        rr_intervals = rr_raw

    time_array = None
    if time_col is not None:
        time_array = sub[time_col].to_numpy(dtype=float)

    valid_mask = np.isfinite(rr_intervals) & (rr_intervals >= 300) & (rr_intervals <= 2000)
    rr_intervals = rr_intervals[valid_mask]
    if time_array is not None:
        time_array = time_array[valid_mask]

    if len(rr_intervals) < 50:
        raise ValueError(f"Слишком мало валидных RR-интервалов ({len(rr_intervals)})")
    return rr_intervals, time_array


def preprocess_rr_intervals(rr_intervals: np.ndarray, min_duration_min: float = 2.0) -> np.ndarray:
    """Предобработка RR-интервалов."""
    duration_min = np.sum(rr_intervals) / 60000.0
    if duration_min < min_duration_min:
        raise ValueError(f"Запись слишком короткая ({duration_min:.2f} мин)")
    return rr_intervals


# =============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# =============================================================================

def safe_fmt(val, fmt=".1f"):
    """Безопасное форматирование для NaN."""
    return f"{val:{fmt}}" if not pd.isna(val) else "N/A"

def plot_hrv_analysis(rr_intervals_original: np.ndarray, rr_intervals_filtered: np.ndarray,
                       title: str, time_column: Optional[np.ndarray] = None,
                       n_removed_outliers: Optional[int] = None) -> plt.Figure:
    """Генерация расширенного сводного графика анализа ВСР.

    n_removed_outliers передаётся из remove_outliers_moving_average()
    (пересчёт через isnan неверен, т.к. выбросы интерполируются).
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontweight='bold', fontsize=14)

    freq_features = compute_frequency_domain_features(rr_intervals_filtered, fs=4.0, time_column=time_column)
    freqs = freq_features['freqs']
    psd = freq_features['psd']
    if not isinstance(freqs, np.ndarray) or not isinstance(psd, np.ndarray):
        freqs = np.asarray([])
        psd = np.asarray([])

    ax1 = plt.subplot(2, 3, 1)
    # Единая временная шкала: валидный time_column используется для обеих кривых;
    # иначе cumsum (время окончания интервалов). Разные шкалы скрывали бы замены.
    if time_column is not None and len(time_column) == len(rr_intervals_filtered) == len(rr_intervals_original):
        cumulative_time_filt = _build_cumulative_time(rr_intervals_filtered, time_column)
        cumulative_time_orig = _build_cumulative_time(rr_intervals_original, time_column)
    else:
        cumulative_time_orig = _build_cumulative_time(np.asarray(rr_intervals_original, dtype=float), None)
        cumulative_time_filt = _build_cumulative_time(np.asarray(rr_intervals_filtered, dtype=float), None)
    cumulative_time_orig_min = cumulative_time_orig / 60.0
    cumulative_time_filt_min = cumulative_time_filt / 60.0

    ax1.plot(cumulative_time_orig_min, rr_intervals_original, color='#adb5bd', linewidth=0.6, alpha=0.5, label='Исходные')
    ax1.plot(cumulative_time_filt_min, rr_intervals_filtered, color='#dc3545', linewidth=0.8, alpha=0.9, label='После фильтрации')
    ax1.set_title("Тахограмма (RR-интервалы)")
    ax1.set_xlabel("Время (мин)"); ax1.set_ylabel("RR (мс)"); ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(rr_intervals_original, bins=50, color='#adb5bd', alpha=0.5, edgecolor='black', label='Исходные', density=True)
    ax2.hist(rr_intervals_filtered, bins=50, color='#0d6efd', alpha=0.6, edgecolor='black', label='После фильтрации', density=True)
    ax2.set_title("Гистограмма RR-интервалов")
    ax2.set_xlabel("RR (мс)"); ax2.set_ylabel("Плотность"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(freqs, psd, color='#6f42c1', linewidth=1.5)
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    if np.any(vlf_mask): ax3.fill_between(freqs[vlf_mask], psd[vlf_mask], alpha=0.3, color='#dc3545', label='VLF')
    if np.any(lf_mask): ax3.fill_between(freqs[lf_mask], psd[lf_mask], alpha=0.3, color='#ffc107', label='LF')
    if np.any(hf_mask): ax3.fill_between(freqs[hf_mask], psd[hf_mask], alpha=0.3, color='#198754', label='HF')
    ax3.set_title("Спектральная плотность мощности (PSD)")
    ax3.set_xlabel("Частота (Гц)"); ax3.set_ylabel("PSD (мс²/Гц)"); ax3.set_xlim(0, 0.5); ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)

    ax4 = plt.subplot(2, 3, 4)
    rr_n = rr_intervals_filtered[:-1]
    rr_n1 = rr_intervals_filtered[1:]
    sc = ax4.scatter(rr_n, rr_n1, c=cumulative_time_filt_min[1:], cmap='viridis', s=15, alpha=0.6, edgecolors='k', linewidth=0.2)

    # ИСПРАВЛЕНИЕ 3: Добавлено ddof=1 для графика Пуанкаре
    sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2) if len(rr_n) > 0 else float('nan')
    sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2) if len(rr_n) > 0 else float('nan')

    min_val = min(rr_n.min(), rr_n1.min()) if len(rr_n) > 0 else 0
    max_val = max(rr_n.max(), rr_n1.max()) if len(rr_n) > 0 else 1
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')

    if not pd.isna(sd1) and not pd.isna(sd2):
        mean_rr_n = np.mean(rr_n)
        mean_rr_n1 = np.mean(rr_n1)
        theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = mean_rr_n + sd2 * np.cos(np.pi/4) * np.cos(theta) - sd1 * np.sin(np.pi/4) * np.sin(theta)
        ellipse_y = mean_rr_n1 + sd2 * np.cos(np.pi/4) * np.sin(theta) + sd1 * np.sin(np.pi/4) * np.cos(theta)
        ax4.plot(ellipse_x, ellipse_y, 'r-', linewidth=2, alpha=0.7, label=f'SD1={safe_fmt(sd1)}, SD2={safe_fmt(sd2)}')

    ax4.set_title("Poincar Plot (SD1/SD2)")
    ax4.set_xlabel("RR_n (мс)"); ax4.set_ylabel("RR_n+1 (мс)"); ax4.set_aspect('equal'); ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=9)
    plt.colorbar(sc, ax=ax4, label="Время (мин)")

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    features = compute_all_hrv_features(rr_intervals_filtered, time_column=time_column)
    stats_text = (
        f"Значения ВСР:\n\n"
        f"Временные:\n"
        f"• Mean RR: {safe_fmt(features['mean_rr_ms'])} мс\n"
        f"• SDNN: {safe_fmt(features['sdnn_ms'])} мс\n"
        f"• RMSSD: {safe_fmt(features['rmssd_ms'])} мс\n"
        f"• pNN50: {safe_fmt(features['pnn50_percent'])}%\n\n"
        f"Спектральные:\n"
        f"• VLF: {safe_fmt(features['vlf_ms2'])} мс²\n"
        f"• LF: {safe_fmt(features['lf_ms2'])} мс²\n"
        f"• HF: {safe_fmt(features['hf_ms2'])} мс²\n"
        f"• LF/HF: {safe_fmt(features['lf_hf_ratio'], '.2f')}\n\n"
        f"Нелинейные:\n"
        f"• Sample Entropy: {safe_fmt(features['sample_entropy'], '.3f')}\n"
        f"• ApEn: {safe_fmt(features.get('approx_entropy'), '.3f')}\n"
        f"• FuzzyEn: {safe_fmt(features.get('fuzzy_entropy'), '.3f')}\n"
        f"• DFA 1: {safe_fmt(features['dfa_alpha1'], '.3f')}\n"
        f"• DFA 2: {safe_fmt(features.get('dfa_alpha2'), '.3f')}\n"
        f"• MSE mean: {safe_fmt(features.get('mse_mean'), '.3f')}\n"
        f"• MSE slope: {safe_fmt(features.get('mse_slope'), '.3f')}\n\n"
        f"Пуанкаре:\n"
        f"• SD1: {safe_fmt(sd1)} мс\n"
        f"• SD2: {safe_fmt(sd2)} мс\n\n"
        f"Индексы:\n"
        f"• ИН: {safe_fmt(features['in_index'], '.4f')}\n"
        f"• ПАПР: {safe_fmt(features['papr'], '.4f')}\n"
        f"• ВПР: {safe_fmt(features['vpr'], '.4f')}"
    )
    ax5.text(0.1, 0.5, stats_text, ha='left', va='center', bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=0.8), fontsize=9, family='monospace')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    if n_removed_outliers is None:
        n_removed_str = "N/A (передайте из remove_outliers_moving_average)"
    else:
        n_removed_str = str(int(n_removed_outliers))
    filter_info = (
        f"Параметры анализа:\n\n"
        f"• Исходных RR: {len(rr_intervals_original)}\n"
        f"• После фильтрации: {len(rr_intervals_filtered)}\n"
        f"• Заменено выбросов: {n_removed_str}\n"
        f"• Длительность: {safe_fmt(features['duration_min'], '.2f')} мин\n"
        f"• Δf: {safe_fmt(freq_features.get('freq_resolution_hz'), '.4f')} Гц\n\n"
        f"Частотные диапазоны:\n"
        f"• VLF: 0.003-0.04 Гц\n"
        f"• LF: 0.04-0.15 Гц\n"
        f"• HF: 0.15-0.4 Гц"
    )
    ax6.text(0.1, 0.5, filter_info, ha='left', va='center', bbox=dict(boxstyle='round,pad=1', facecolor='#e7f3ff', alpha=0.8), fontsize=9, family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# =============================================================================
# 7. ГЕНЕРАЦИЯ ОТЧЕТОВ (HTML и XLSX)
# =============================================================================

def build_results_table_vertical(features: Dict[str, float], norms: Dict[str, Dict[str, Any]]) -> str:
    """Строит вертикальную HTML-таблицу результатов анализа."""
    display_order = [
        "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_percent",
        "lf_ms2", "hf_ms2", "lf_hf_ratio",
        "sample_entropy", "approx_entropy", "fuzzy_entropy",
        "dfa_alpha1", "dfa_alpha2", "mse_mean", "mse_slope",
        "sd1_ms", "sd2_ms",
        "in_index", "papr", "vpr"
    ]

    rows_html = ""
    for metric in display_order:
        if metric not in features or metric not in norms: continue

        value = features[metric]
        norm = norms[metric]
        desc = norm.get("desc", metric)
        unit = norm.get("unit", "")

        interp = get_indicator_interpretation(value, metric, norms)

        if pd.isna(value):
            value_str = "Нет данных"
            unit_str = ""
        else:
            unit_str = f" {unit}" if unit else ""
            value_str = f"{value:.3f}{unit_str}"

        font_color = "#000000" if interp["color"] == "#ffc107" else "#ffffff"
        rows_html += f"""
        <tr>
            <td class="metric-name">{desc}</td>
            <td class="metric-value">{value_str}</td>
            <td class="metric-interp" style="background-color: {interp['color']}; color: {font_color};"
                title="{interp['direction']}">{interp['interpretation']}</td>
        </tr>
        """
    return rows_html


def save_html_report(features_list: List[Dict[str, float]], plots: Dict[str, plt.Figure],
                     out_path: Path, norms: Dict[str, Dict[str, Any]]):
    """Создание HTML-отчета."""
    ZOOM_MODAL_CSS = (
        "<style>"
        ".modal { background: rgba(0,0,0,0.94); }"
        ".modal-content { width: 96vw; max-width: 2000px; box-shadow: 0 0 40px rgba(0,0,0,0.6); }"
        ".modal-content img { width: 100%; height: auto; }"
        ".close-modal { font-size: 2.5rem; }"
        ".block-title { color: #dc3545; border-bottom: 2px solid #dc3545; padding-bottom: 0.3rem; margin: 1.5rem 0 0.4rem; font-size: 1.2rem; }"
        ".test-subtitle { color: #0d6efd; margin: 0 0 0.8rem; font-size: 1rem; font-weight: 600; }"
        "</style>"
    )
    builder = HTMLReportBuilder(title=" Отчет по анализу вариабельности сердечного ритма")
    builder.add_html(ZOOM_MODAL_CSS, as_section=False)

    for features in features_list:
        test_id = features.get("test_id", "unknown")
        rows_html = build_results_table_vertical(features, norms)
        block = (
            f'<div class="result-block">'
            f'<table class="results-table">'
            f'<thead><tr><th>Показатель</th><th>Значение</th><th>Интерпретация</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
            f'</div>'
        )
        builder.add_html(block, as_section=True, title=test_id)

    for test_id, fig in plots.items():
        builder.add_figure(fig, title="Визуализация")

    builder.save(str(out_path))
    print(f" HTML-отчет сохранен: {out_path}")


def save_xlsx_report(features_list: List[Dict[str, float]], out_path: Path, norms: Dict[str, Dict[str, Any]]):
    """Сохранение результатов в Excel."""
    display_order = [
        "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_percent",
        "lf_ms2", "hf_ms2", "lf_hf_ratio",
        "sample_entropy", "approx_entropy", "fuzzy_entropy",
        "dfa_alpha1", "dfa_alpha2", "mse_mean", "mse_slope",
        "sd1_ms", "sd2_ms",
        "in_index", "papr", "vpr"
    ]

    rows = []
    interp_colors = []
    for features in features_list:
        test_id = features.get("test_id", "unknown")
        for metric in display_order:
            if metric not in features or metric not in norms: continue
            value = features[metric]
            norm = norms[metric]
            interp = get_indicator_interpretation(value, metric, norms)

            if pd.isna(value):
                val_str = "Нет данных"
            else:
                val_str = str(round(float(value), 3))

            rows.append({
                "Тест": test_id,
                "Показатель": norm.get("desc", metric),
                "Значение": val_str,
                "Ед.изм.": norm.get("unit", ""),
                "Интерпретация": interp["interpretation"]
            })
            interp_colors.append(interp["color"])

    df = pd.DataFrame(rows)
    if df.empty:
        print(f" Нет данных для сохранения в Excel: {out_path}")
        return

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Результаты ВСР', index=False)
        worksheet = writer.sheets['Результаты ВСР']
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df[col]) > 0 else 0, len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

        interp_col = df.columns.get_loc("Интерпретация") + 1
        interp_letter = chr(64 + interp_col) if interp_col <= 26 else chr(64 + (interp_col - 1) // 26) + chr(64 + (interp_col - 1) % 26 + 1)
        for row_idx, color in enumerate(interp_colors, start=2):
            cell = worksheet[f"{interp_letter}{row_idx}"]
            font_color = "000000" if color == "#ffc107" else "FFFFFF"
            cell.fill = PatternFill(start_color=color.lstrip("#"), end_color=color.lstrip("#"), fill_type="solid")
            cell.font = Font(color=font_color, bold=True)

    print(f" XLSX-отчет сохранен: {out_path}")


# =============================================================================
# 8. ПОИСК ФАЙЛОВ
# =============================================================================

def discover_data_files(root_dir: Path) -> List[Dict]:
    """Поиск CSV и TXT файлов с RR-интервалами."""
    data_files = []
    for ext in ['*.csv', '*.txt']:
        for f in root_dir.rglob(ext):
            try:
                for delimiter in [',', ';', '\t', ' ', None]:
                    try:
                        df_test = pd.read_csv(f, delimiter=delimiter, nrows=5)
                        if not df_test.empty:
                            numeric_cols = df_test.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                data_files.append({"id": f.stem, "path": f})
                                break
                    except: continue
            except: pass
    return data_files
