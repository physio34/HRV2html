#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💓 Универсальный анализатор вариабельности сердечного ритма (ВСР)
Поддерживает RR-интервалы из CSV и TXT файлов.
Расчет спектральных, энтропийных, фрактальных показателей и индексов ИН, ПАПР, ВПР.
Фильтрация выбросов методом скользящего среднего.
Генерирует HTML и XLSX отчеты с цветовой индикацией "светофор".
Нормативы загружаются из внешнего конфигурационного файла hrv_norms.json.

УЛУЧШЕНИЯ ПО СРАВНЕНИЮ С analys_html_individual.py:
  * compute_sample_entropy переписана через scipy.spatial.KDTree
    (при наличии библиотеки antropy используется она как эталон).
  * Добавлен расширенный энтропийный анализ из analys_html_group.py:
    ApEn, FuzzyEn, DFA α1/α2, мультимасштабная энтропия (MSE mean/slope)
    с условиями применимости по длине записи.
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
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.spatial import cKDTree as KDTree

__version__ = "1.0.0"

# Настройки matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
pd.options.mode.chained_assignment = None


# =============================================================================
# 1. МАТЕМАТИЧЕСКОЕ ЯДРО (Дескрипторы ВСР)
# =============================================================================

def compute_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Вычисляет временные показатели ВСР."""
    rr_diff = np.diff(rr_intervals)
    rr_diff_abs = np.abs(rr_diff)
    
    sdnn = float(np.std(rr_intervals))
    rmssd = float(np.sqrt(np.mean(rr_diff**2)))
    pnn50 = float(np.sum(rr_diff_abs > 50) / len(rr_diff) * 100)
    mean_rr = float(np.mean(rr_intervals))
    
    duration_sec = np.sum(rr_intervals) / 1000.0
    if duration_sec >= 300:
        n_segments = int(duration_sec // 300)
        segment_means = []
        for i in range(n_segments):
            start_idx = int(i * 300 * 1000 / mean_rr)
            end_idx = int((i + 1) * 300 * 1000 / mean_rr)
            if end_idx <= len(rr_intervals):
                segment_means.append(np.mean(rr_intervals[start_idx:end_idx]))
        sdann = float(np.std(segment_means)) if segment_means else 0.0
    else:
        sdann = 0.0
    
    return {
        "mean_rr_ms": round(mean_rr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50_percent": round(pnn50, 2),
        "sdann_ms": round(sdann, 2)
    }


def compute_frequency_domain_features(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    """Вычисляет спектральные показатели ВСР методом Уэлча."""
    rr_detrended = rr_intervals - np.mean(rr_intervals)
    
    cumulative_time = cumulative_trapezoid(rr_intervals / 1000.0, initial=0)
    t_uniform = np.arange(0, cumulative_time[-1], 1.0 / fs)
    rr_uniform = np.interp(t_uniform, cumulative_time, rr_detrended)
    
    nperseg = min(256, len(rr_uniform))
    freqs, psd = signal.welch(rr_uniform, fs=fs, nperseg=nperseg, scaling='density')
    
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    
    vlf = float(trapezoid(psd[vlf_mask], x=freqs[vlf_mask])) if np.any(vlf_mask) else 0.0
    lf = float(trapezoid(psd[lf_mask], x=freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf = float(trapezoid(psd[hf_mask], x=freqs[hf_mask])) if np.any(hf_mask) else 0.0
    
    total_power = vlf + lf + hf
    lf_hf_ratio = lf / hf if hf > 0 else 0.0
    
    return {
        "vlf_ms2": round(vlf, 2),
        "lf_ms2": round(lf, 2),
        "hf_ms2": round(hf, 2),
        "total_power_ms2": round(total_power, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 3),
        "lf_norm": round(lf / (total_power - vlf) * 100 if (total_power - vlf) > 0 else 0.0, 2),
        "hf_norm": round(hf / (total_power - vlf) * 100 if (total_power - vlf) > 0 else 0.0, 2),
        "freqs": freqs,
        "psd": psd
    }


def compute_spectral_peak_features(rr_intervals: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    """Пиковые частоты спектральных полос и нормированные доли (nu).

    Возвращает:
      * peak_vlf_hz, peak_lf_hz, peak_hf_hz — частоты пиков PSD в полосах
        VLF (0.003–0.04), LF (0.04–0.15), HF (0.15–0.4) Гц;
      * fraction_lf_nu, fraction_hf_nu — нормированные единицы (nu),
        равные lf_norm / hf_norm из спектрального анализа.
    """
    freq_features = compute_frequency_domain_features(rr_intervals, fs=fs)
    freqs = freq_features["freqs"]
    psd = freq_features["psd"]

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
    """Вычисляет Sample Entropy (SampEn).

    Реализация через scipy.spatial.KDTree: шаблоны длины m и m+1
    индексируются в дереве, подсчёт совпадений (Chebyshev-расстояние <= r*std)
    выполняется векторизованным запросом соседей. При установленной
    библиотеке antropy используется её (валидированная) реализация.
    """
    # Эталонная реализация, если доступна antropy.
    try:
        import antropy as _ant
        return float(_ant.sample_entropy(np.asarray(rr_intervals, dtype=float),
                                         order=m, r=r))
    except Exception:
        pass

    rr = np.asarray(rr_intervals, dtype=float)
    n = len(rr)
    if n < m + 2:
        return 0.0

    threshold = float(r) * float(np.std(rr))
    if threshold <= 0:
        return 0.0

    def count_matches(template_length: int) -> float:
        pats = np.array([rr[i:i + template_length]
                         for i in range(n - template_length + 1)])
        # SampEn использует расстояние Чебышёва (максимум модуля разности),
        # что соответствует норме Минковского p=inf в query_ball_point.
        tree = KDTree(pats)
        # query_ball_point возвращает для каждой точки список соседей
        # (включая саму точку); вычитаем 1, чтобы исключить самосовпадение.
        neighbours = tree.query_ball_point(pats, threshold, p=np.inf)
        return float(sum(len(x) - 1 for x in neighbours))

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def _dfa_entropy(rr: np.ndarray, scales_short=(4, 16), scales_long=(17, 64)) -> Tuple[float, float]:
    """DFA α1 (короткие шкалы) и α2 (длинные шкалы). Адаптировано из analys_html_group.py."""
    try:
        y = np.cumsum(rr - np.mean(rr))
        n = len(y)

        def calc_alpha(scales):
            fluct = []
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
            return np.polyfit(np.log(scales[:len(fluct)]), np.log(fluct), 1)[0] \
                if len(fluct) >= 2 else np.nan

        return (calc_alpha(np.arange(scales_short[0], scales_short[1] + 1, dtype=int)),
                calc_alpha(np.arange(scales_long[0], scales_long[1] + 1, dtype=int)))
    except Exception:
        return np.nan, np.nan


def _approx_entropy(u: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate Entropy (ApEn). Адаптировано из analys_html_group.py."""
    if len(u) < m + 2:
        return np.nan
    N = len(u)
    r = float(r) * np.std(u)

    def _phi(mm):
        x = np.array([u[i:i + mm] for i in range(N - mm + 1)])
        C = np.mean([np.sum(np.max(np.abs(x - xi), axis=1) <= r) for xi in x]) / (N - mm + 1)
        return np.mean(np.log(np.maximum(C, 1e-9)))

    return float(_phi(m) - _phi(m + 1))


def _fuzzy_entropy(u: np.ndarray, m: int = 2, r: float = 0.15, n: float = 2.0) -> float:
    """Fuzzy Entropy (FuzzyEn). Адаптировано из analys_html_group.py."""
    if len(u) < m + 2:
        return np.nan
    N = len(u)
    r = float(r) * np.std(u)

    def _phi(mm):
        x = np.array([u[i:i + mm] for i in range(N - mm + 1)])
        dist = np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
        np.fill_diagonal(dist, np.inf)
        mu = np.exp(-(dist ** n) / (r + 1e-9))
        return np.sum(mu) / ((N - mm + 1) * (N - mm))

    A, B = _phi(m + 1), _phi(m)
    return float(-np.log(A / (B + 1e-9))) if B > 0 else np.nan


def _multiscale_entropy(rr: np.ndarray, max_scale: int = 20, m: int = 2,
                        r: float = 0.15) -> Dict[str, float]:
    """Мультимасштабная энтропия (MSE). Адаптировано из analys_html_group.py."""
    entropies = []
    for scale in range(1, max_scale + 1):
        cg_rr = np.array([np.mean(rr[i:i + scale])
                         for i in range(0, len(rr) - scale + 1, scale)])
        if len(cg_rr) < m + 2:
            entropies.append(np.nan)
            continue
        entropies.append(compute_sample_entropy(cg_rr, m=m, r=r))
    entropies = np.array(entropies)
    valid = entropies[~np.isnan(entropies)]
    if len(valid) < 2:
        return {"mse_mean": np.nan, "mse_slope": np.nan}
    return {
        "mse_mean": float(np.mean(valid)),
        "mse_slope": float(np.polyfit(np.arange(1, len(valid) + 1), valid, 1)[0]),
    }


def compute_entropy_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Расширенный энтропийный/фрактальный анализ с учётом условий применимости.

    Возвращает: sample_entropy, approx_entropy, fuzzy_entropy,
    dfa_alpha1, dfa_alpha2, mse_mean, mse_slope. Метрики, требующие
    большей длины записи, заполняются 0.0, если условие не выполнено.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    n = len(rr)
    feats: Dict[str, float] = {}

    # Sample Entropy (KDTree / antropy) — основная метрика сложности.
    try:
        feats["sample_entropy"] = round(compute_sample_entropy(rr, m=2, r=0.2), 3)
    except Exception:
        feats["sample_entropy"] = 0.0

    # DFA требует >= 64 точек.
    if n >= 64:
        a1, a2 = _dfa_entropy(rr)
        feats["dfa_alpha1"] = round(float(a1), 3) if np.isfinite(a1) else 0.0
        feats["dfa_alpha2"] = round(float(a2), 3) if np.isfinite(a2) else 0.0
    else:
        feats["dfa_alpha1"] = 0.0
        feats["dfa_alpha2"] = 0.0

    # ApEn / FuzzyEn требуют >= 150 точек.
    if n >= 150:
        rr_norm = (rr - np.mean(rr)) / (np.std(rr) + 1e-10)
        try:
            feats["approx_entropy"] = round(_approx_entropy(rr_norm, m=2, r=0.2), 3)
        except Exception:
            feats["approx_entropy"] = 0.0
        try:
            feats["fuzzy_entropy"] = round(_fuzzy_entropy(rr_norm, m=2, r=0.15, n=2.0), 3)
        except Exception:
            feats["fuzzy_entropy"] = 0.0
    else:
        pass  # ключи добавим как 0.0 ниже для согласованности отчёта

    # MSE требует >= 500 точек (длинная качественная запись).
    if n >= 500:
        try:
            mse = _multiscale_entropy(rr, max_scale=20, m=2, r=0.15)
            feats["mse_mean"] = round(mse["mse_mean"], 3) if np.isfinite(mse["mse_mean"]) else 0.0
            feats["mse_slope"] = round(mse["mse_slope"], 3) if np.isfinite(mse["mse_slope"]) else 0.0
        except Exception:
            feats["mse_mean"] = 0.0
            feats["mse_slope"] = 0.0

    for key in ["approx_entropy", "fuzzy_entropy", "mse_mean", "mse_slope"]:
        feats.setdefault(key, 0.0)
    return feats


def compute_dfa_alpha1(rr_intervals: np.ndarray) -> float:
    """Показатель фрактальности DFA α1 (обёртка над _dfa_entropy)."""
    a1, _ = _dfa_entropy(np.asarray(rr_intervals, dtype=float))
    return float(a1) if np.isfinite(a1) else 0.0


def compute_baevsky_indices(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет классические индексы Баевского: ИН, ПАПР, ВПР.
    Использует фиксированный шаг бина 50 мс (стандарт вариационной пульсометрии).
    """
    if len(rr_intervals) == 0:
        return {"in_index": 0.0, "papr": 0.0, "vpr": 0.0, "mo_ms": 0.0, "amo_percent": 0.0}

    # 1. Определяем Xmax и Xmin (размах вариаций)
    x_max = float(np.max(rr_intervals))
    x_min = float(np.min(rr_intervals))
    range_rr = x_max - x_min
    
    if range_rr == 0:
        return {"in_index": 0.0, "papr": 0.0, "vpr": 0.0, "mo_ms": x_max, "amo_percent": 100.0}

    # 2. Строим гистограмму с ФИКСИРОВАННЫМ шагом 50 мс
    bin_width = 50.0
    
    # Находим границы бинов, чтобы покрыть весь диапазон данных
    min_bin_edge = np.floor(x_min / bin_width) * bin_width
    max_bin_edge = np.ceil(x_max / bin_width) * bin_width
    bins = np.arange(min_bin_edge, max_bin_edge + bin_width, bin_width)
    
    counts, bin_edges = np.histogram(rr_intervals, bins=bins)
    
    # 3. Находим моду (Mo) и АМо (AMo)
    mode_idx = np.argmax(counts)
    
    # В классике чаще используют центр бина. Используем центр для большей точности.
    mo = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0
    
    # АМо - процент интервалов, попавших в модальный бин
    amo_percent = (counts[mode_idx] / len(rr_intervals)) * 100.0
    
    # 4. Вычисляем классические индексы Баевского
    # Переводим в секунды для классической размерности ИН
    mo_sec = mo / 1000.0
    range_sec = range_rr / 1000.0
    
    # ИН = АМо(%) / (2 * Mo(с) * (Xmax - Xmin)(с))
    if mo_sec > 0 and range_sec > 0:
        in_index = amo_percent / (2 * mo_sec * range_sec)
    else:
        in_index = 0.0
        
    # ПАПР = АМо(%) / Mo(мс)  (В C-коде: 1000 * AMo_abs / Mo, что эквивалентно AMo_% / Mo при нормировке)
    papr = amo_percent / mo if mo > 0 else 0.0
    
    # ВПР = (Xmax - Xmin) / Mo  (Классический показатель размаха относительно моды)
    vpr = range_rr / mo if mo > 0 else 0.0
    
    return {
        "in_index": round(in_index, 4),
        "papr": round(papr, 4),
        "vpr": round(vpr, 4),
        "mo_ms": round(mo, 2),
        "amo_percent": round(amo_percent, 2),
        "dx_ms": round(range_rr, 2) # Возвращаем реальный размах, а не ширину бина
    }

def compute_all_hrv_features(rr_intervals: np.ndarray) -> Dict[str, float]:
    """Вычисляет все дескрипторы ВСР."""
    features = {}
    features.update(compute_time_domain_features(rr_intervals))
    
    freq_features = compute_frequency_domain_features(rr_intervals)
    features.update({k: v for k, v in freq_features.items() if k not in ['freqs', 'psd']})

    # Расширенный энтропийный анализ (SampEn/KDTree, ApEn, FuzzyEn, DFA α1/α2, MSE).
    features.update(compute_entropy_features(rr_intervals))

    features.update(compute_baevsky_indices(rr_intervals))
    
    features["n_rr_intervals"] = len(rr_intervals)
    features["duration_min"] = round(np.sum(rr_intervals) / 60000.0, 2)
    
    return features


# =============================================================================
# 2. ФИЛЬТРАЦИЯ ВЫБРОСОВ МЕТОДОМ СКОЛЬЗЯЩЕГО СРЕДНЕГО
# =============================================================================

def remove_outliers_moving_average(rr_intervals: np.ndarray, window_size: int = 5, threshold: float = 3.0) -> Tuple[np.ndarray, int]:
    """
    Удаляет выбросы методом скользящего среднего.
    """
    if len(rr_intervals) < window_size:
        return rr_intervals, 0
    
    kernel = np.ones(window_size) / window_size
    moving_avg = np.convolve(rr_intervals, kernel, mode='same')
    
    for i in range(window_size // 2):
        moving_avg[i] = np.mean(rr_intervals[:i + window_size // 2 + 1])
        moving_avg[-(i+1)] = np.mean(rr_intervals[-(i + window_size // 2 + 1):])
    
    deviations = np.abs(rr_intervals - moving_avg)
    std_dev = np.std(deviations)
    
    if std_dev == 0:
        return rr_intervals, 0
    
    valid_mask = deviations < (threshold * std_dev)
    filtered_rr = rr_intervals[valid_mask]
    n_removed = len(rr_intervals) - len(filtered_rr)
    
    return filtered_rr, n_removed


# =============================================================================
# 3. ЗАГРУЗКА НОРМАТИВОВ ИЗ КОНФИГА
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "hrv_norms.json"

DEFAULT_NORMS: Dict[str, Dict[str, Any]] = {
    "mean_rr_ms": {
        "desc": "Средний RR-интервал",
        "unit": "мс",
        "green": [600, 1200],
        "yellow": [500, 1400],
        "category": "rhythm",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренная тахикардия",
                    "severe": "Значительно ниже нормы. Выраженная тахикардия"},
            "high": {"moderate": "Выше нормы. Умеренная брадикардия",
                     "severe": "Значительно выше нормы. Выраженная брадикардия"}
        }
    },
    "sdnn_ms": {
        "desc": "SDNN (общая вариабельность)",
        "unit": "мс",
        "green": [50, 200],
        "yellow": [30, 50],
        "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение общей ВСР",
                    "severe": "Значительно ниже нормы. Выраженное снижение ВСР"},
            "high": {"moderate": "Выше нормы. Умеренное повышение ВСР",
                     "severe": "Значительно выше нормы. Выраженное повышение ВСР"}
        }
    },
    "rmssd_ms": {
        "desc": "RMSSD (парасимпатическая активность)",
        "unit": "мс",
        "green": [20, 100],
        "yellow": [10, 20],
        "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение вагусного влияния",
                    "severe": "Значительно ниже нормы. Выраженная симпатикотония"},
            "high": {"moderate": "Выше нормы. Умеренная ваготония",
                     "severe": "Значительно выше нормы. Выраженная ваготония"}
        }
    },
    "pnn50_percent": {
        "desc": "pNN50 (парасимпатический индекс)",
        "unit": "%",
        "green": [3, 20],
        "yellow": [1, 3],
        "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренная симпатикотония",
                    "severe": "Значительно ниже нормы. Выраженная симпатикотония"},
            "high": {"moderate": "Выше нормы. Умеренная ваготония",
                     "severe": "Значительно выше нормы. Выраженная ваготония"}
        }
    },
    "lf_hf_ratio": {
        "desc": "LF/HF (баланс симпатико-вагусных влияний)",
        "unit": "",
        "green": [1.0, 2.0],
        "yellow": [0.5, 3.0],
        "category": "balance",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренная ваготония",
                    "severe": "Значительно ниже нормы. Выраженная ваготония"},
            "high": {"moderate": "Выше нормы. Умеренная симпатикотония",
                     "severe": "Значительно выше нормы. Выраженная симпатикотония"}
        }
    },
    "hf_ms2": {
        "desc": "HF мощность (вагусный контроль)",
        "unit": "мс²",
        "green": [500, 3000],
        "yellow": [200, 500],
        "category": "parasympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение вагусного влияния",
                    "severe": "Значительно ниже нормы. Выраженная симпатизация"},
            "high": {"moderate": "Выше нормы. Умеренная ваготония",
                     "severe": "Значительно выше нормы. Выраженная ваготония"}
        }
    },
    "lf_ms2": {
        "desc": "LF мощность (симпатический контроль)",
        "unit": "мс²",
        "green": [300, 2000],
        "yellow": [100, 300],
        "category": "sympathetic",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение симпатического влияния",
                    "severe": "Значительно ниже нормы. Выраженная ваготония"},
            "high": {"moderate": "Выше нормы. Умеренная симпатикотония",
                     "severe": "Значительно выше нормы. Выраженная симпатикотония"}
        }
    },
    "in_index": {
        "desc": "ИН Баевского (индекс напряжения)",
        "unit": "",
        "green": [50, 150],
        "yellow": [25, 300],
        "category": "stress",
        "interpretation": {
            "normal": "Норма. Удовлетворительная адаптация",
            "low": {"moderate": "Ниже нормы. Умеренное снижение напряжения РС",
                    "severe": "Значительно ниже нормы. Истощение регуляторных систем"},
            "high": {"moderate": "Выше нормы. Умеренное напряжение регуляторных систем",
                     "severe": "Значительно выше нормы. Выраженное напряжение РС"}
        }
    },
    "papr": {
        "desc": "ПАПР (адекватность регуляции)",
        "unit": "",
        "green": [1.0, 3.0],
        "yellow": [0.5, 1.0],
        "category": "regulation",
        "interpretation": {
            "normal": "Норма. Адекватная регуляция",
            "low": {"moderate": "Ниже нормы. Умеренное снижение адекватности регуляции",
                    "severe": "Значительно ниже нормы. Выраженная дезадаптация"},
            "high": {"moderate": "Выше нормы. Умеренное перенапряжение регуляции",
                     "severe": "Значительно выше нормы. Выраженная дезадаптация"}
        }
    },
    "vpr": {
        "desc": "ВПР (вегетативный индекс реактивности)",
        "unit": "",
        "green": [1.0, 3.0],
        "yellow": [0.5, 1.0],
        "category": "reactivity",
        "interpretation": {
            "normal": "Норма. Адекватная реактивность",
            "low": {"moderate": "Ниже нормы. Умеренное снижение реактивности",
                    "severe": "Значительно ниже нормы. Выраженное снижение реактивности"},
            "high": {"moderate": "Выше нормы. Умеренное повышение реактивности",
                     "severe": "Значительно выше нормы. Гиперреактивность"}
        }
    },
    "sample_entropy": {
        "desc": "Sample Entropy (сложность ритма)",
        "unit": "",
        "green": [1.0, 2.5],
        "yellow": [0.5, 1.0],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма",
                    "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма",
                     "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "dfa_alpha1": {
        "desc": "DFA α1 (фрактальная корреляция)",
        "unit": "",
        "green": [1.0, 1.5],
        "yellow": [0.75, 1.0],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая фрактальная структура",
            "low": {"moderate": "Ниже нормы. Умеренная случайность ритма",
                    "severe": "Значительно ниже нормы. Выраженная хаотизация"},
            "high": {"moderate": "Выше нормы. Умеренная избыточная регулярность",
                     "severe": "Значительно выше нормы. Избыточная предсказуемость"}
        }
    },
    "sd1_ms": {
        "desc": "SD1 (краткосрочная вариабельность)",
        "unit": "мс",
        "green": [10, 50],
        "yellow": [5, 10],
        "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение краткосрочной вариабельности",
                    "severe": "Значительно ниже нормы. Выраженное снижение"},
            "high": {"moderate": "Выше нормы. Умеренное повышение краткосрочной вариабельности",
                     "severe": "Значительно выше нормы. Выраженное повышение"}
        }
    },
    "sd2_ms": {
        "desc": "SD2 (долгосрочная вариабельность)",
        "unit": "мс",
        "green": [30, 150],
        "yellow": [20, 30],
        "category": "variability",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное снижение долгосрочной вариабельности",
                    "severe": "Значительно ниже нормы. Выраженное снижение"},
            "high": {"moderate": "Выше нормы. Умеренное повышение долгосрочной вариабельности",
                     "severe": "Значительно выше нормы. Выраженное повышение"}
        }
    },
    "dfa_alpha2": {
        "desc": "DFA α2 (фрактальная корреляция, длинные шкалы)",
        "unit": "",
        "green": [0.8, 1.2],
        "yellow": [0.6, 0.8],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая фрактальная структура на длинных шкалах",
            "low": {"moderate": "Ниже нормы. Умеренная случайность на длинных шкалах",
                    "severe": "Значительно ниже нормы. Выраженная хаотизация"},
            "high": {"moderate": "Выше нормы. Умеренная избыточная регулярность",
                     "severe": "Значительно выше нормы. Избыточная предсказуемость"}
        }
    },
    "approx_entropy": {
        "desc": "Approximate Entropy (регулярность ритма)",
        "unit": "",
        "green": [0.8, 1.8],
        "yellow": [0.4, 0.8],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма",
                    "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма",
                     "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "fuzzy_entropy": {
        "desc": "Fuzzy Entropy (мягкая нерегулярность ритма)",
        "unit": "",
        "green": [0.5, 1.5],
        "yellow": [0.2, 0.5],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение структуры ритма",
                    "severe": "Значительно ниже нормы. Выраженное упрощение (ригидность)"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение ритма",
                     "severe": "Значительно выше нормы. Хаотизация ритма"}
        }
    },
    "mse_mean": {
        "desc": "MSE mean (средняя мультимасштабная энтропия)",
        "unit": "",
        "green": [0.8, 1.8],
        "yellow": [0.4, 0.8],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма. Здоровая регуляторная сложность по масштабам",
            "low": {"moderate": "Ниже нормы. Умеренное упрощение на мультимасштабе",
                    "severe": "Значительно ниже нормы. Выраженная ригидность"},
            "high": {"moderate": "Выше нормы. Умеренное усложнение",
                     "severe": "Значительно выше нормы. Хаотизация"}
        }
    },
    "mse_slope": {
        "desc": "MSE slope (наклон мультимасштабной энтропии)",
        "unit": "",
        "green": [-0.05, 0.05],
        "yellow": [-0.1, -0.05],
        "category": "complexity",
        "interpretation": {
            "normal": "Норма. Близкий к нулю наклон (здоровая адаптация)",
            "low": {"moderate": "Отрицательный наклон. Снижение сложности с масштабом",
                    "severe": "Выраженный отрицательный наклон (признак патологии)"},
            "high": {"moderate": "Положительный наклон. Рост сложности с масштабом",
                     "severe": "Выраженный положительный наклон"}
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
            print(f"✅ Нормативы загружены из: {config_path.name}")
            return data
        except Exception as e:
            print(f"⚠️ Ошибка чтения конфига {config_path}: {e}. Используются встроенные нормативы.")
            return DEFAULT_NORMS
    else:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_NORMS, f, ensure_ascii=False, indent=2)
            print(f"📝 Создан файл нормативов: {config_path.name} (отредактируйте при необходимости)")
        except Exception as e:
            print(f"⚠️ Не удалось создать файл конфига: {e}")
        return DEFAULT_NORMS


# =============================================================================
# 4. ИНТЕРПРЕТАЦИЯ ПОКАЗАТЕЛЕЙ (СИСТЕМА "СВЕТОФОР")
# =============================================================================

def get_indicator_interpretation(value: float, metric_name: str, norms: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Определяет цвет, статус и текстовую интерпретацию для показателя ВСР."""
    if metric_name not in norms:
        return {
            "color": "#6c757d",
            "status": "—",
            "interpretation": "Показатель не имеет норматива",
            "direction": ""
        }
    
    norm = norms[metric_name]
    green = norm["green"]
    yellow = norm["yellow"]
    interp = norm.get("interpretation", {})
    
    if green[0] <= value <= green[1]:
        return {
            "color": "#198754",
            "status": "Норма",
            "interpretation": interp.get("normal", "Норма"),
            "direction": "в пределах нормы"
        }
    
    if value < green[0]:
        direction = "ниже нормы"
        severity_key = "low"
    else:
        direction = "выше нормы"
        severity_key = "high"
    
    if yellow[0] <= value <= yellow[1] or (value < green[0] and value >= yellow[0]) or (value > green[1] and value <= yellow[1]):
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
    """Автоопределение столбца с RR-интервалами."""
    cols_lower = {c: str(c).lower().strip() for c in df.columns}
    rr_patterns = ['rr', 'nn', 'rr_interval', 'nn_interval', 'rri', 'r-r', 'n-n',
                   'interbeat', 'ibi', 'interval', 'ms']
    
    rr_col = next((c for c, l in cols_lower.items() if any(p in l for p in rr_patterns)), None)
    
    if rr_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            rr_col = numeric_cols[0]
        else:
            raise ValueError("Не найден столбец с RR-интервалами")
    return rr_col


def load_rr_intervals_from_file(filepath: Union[str, Path]) -> np.ndarray:
    """Загрузка RR-интервалов из CSV или TXT файла."""
    filepath = Path(filepath)
    for delimiter in [',', ';', '\t', ' ', None]:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter)
            if not df.empty:
                break
        except:
            continue
    else:
        raise ValueError(f"Не удалось прочитать файл {filepath.name}")
    
    if df.empty:
        raise ValueError(f"Файл {filepath.name} пустой")
    
    rr_col = detect_rr_column(df)
    rr_intervals = df[rr_col].dropna().to_numpy(dtype=float)
    
    valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
    rr_intervals = rr_intervals[valid_mask]
    
    if len(rr_intervals) < 50:
        raise ValueError(f"Слишком мало валидных RR-интервалов ({len(rr_intervals)})")
    return rr_intervals


def preprocess_rr_intervals(rr_intervals: np.ndarray, min_duration_min: float = 2.0) -> np.ndarray:
    """Предобработка RR-интервалов."""
    duration_min = np.sum(rr_intervals) / 60000.0
    if duration_min < min_duration_min:
        raise ValueError(f"Запись слишком короткая ({duration_min:.2f} мин)")
    return rr_intervals


# =============================================================================
# 6. ВИЗУАЛИЗАЦИЯ (РАСШИРЕННАЯ)
# =============================================================================

def plot_hrv_analysis(rr_intervals_original: np.ndarray, rr_intervals_filtered: np.ndarray, title: str) -> plt.Figure:
    """Генерация расширенного сводного графика анализа ВСР с PSD и Poincaré plot."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontweight='bold', fontsize=14)

    # Вычисляем спектральные данные один раз
    freq_features = compute_frequency_domain_features(rr_intervals_filtered, fs=4.0)
    freqs = freq_features['freqs']
    psd = freq_features['psd']

    # 1. Тахограмма (временной ряд RR) - исходные и отфильтрованные
    ax1 = plt.subplot(2, 3, 1)
    cumulative_time_orig = cumulative_trapezoid(rr_intervals_original / 1000.0, initial=0) / 60.0
    cumulative_time_filt = cumulative_trapezoid(rr_intervals_filtered / 1000.0, initial=0) / 60.0

    # Исходные данные (серым)
    ax1.plot(cumulative_time_orig, rr_intervals_original, color='#adb5bd', linewidth=0.6, alpha=0.5, label='Исходные')
    # Отфильтрованные данные (цветным)
    ax1.plot(cumulative_time_filt, rr_intervals_filtered, color='#dc3545', linewidth=0.8, alpha=0.9, label='После фильтрации')

    ax1.set_title("Тахограмма (RR-интервалы)")
    ax1.set_xlabel("Время (мин)")
    ax1.set_ylabel("RR (мс)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Гистограмма RR-интервалов - исходные и отфильтрованные
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(rr_intervals_original, bins=50, color='#adb5bd', alpha=0.5, edgecolor='black', label='Исходные', density=True)
    ax2.hist(rr_intervals_filtered, bins=50, color='#0d6efd', alpha=0.6, edgecolor='black', label='После фильтрации', density=True)
    ax2.set_title("Гистограмма RR-интервалов")
    ax2.set_xlabel("RR (мс)")
    ax2.set_ylabel("Плотность")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Спектральная плотность мощности (PSD)
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(freqs, psd, color='#6f42c1', linewidth=1.5)

    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    if np.any(vlf_mask):
        ax3.fill_between(freqs[vlf_mask], psd[vlf_mask], alpha=0.3, color='#dc3545', label='VLF')
    if np.any(lf_mask):
        ax3.fill_between(freqs[lf_mask], psd[lf_mask], alpha=0.3, color='#ffc107', label='LF')
    if np.any(hf_mask):
        ax3.fill_between(freqs[hf_mask], psd[hf_mask], alpha=0.3, color='#198754', label='HF')

    ax3.set_title("Спектральная плотность мощности (PSD)")
    ax3.set_xlabel("Частота (Гц)")
    ax3.set_ylabel("PSD (мс²/Гц)")
    ax3.set_xlim(0, 0.5)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)

    # 4. Poincaré plot (RR_n vs RR_n+1) с SD1/SD2
    ax4 = plt.subplot(2, 3, 4)
    rr_n = rr_intervals_filtered[:-1]
    rr_n1 = rr_intervals_filtered[1:]

    sc = ax4.scatter(rr_n, rr_n1, c=cumulative_time_filt[1:], cmap='viridis', s=15, alpha=0.6, edgecolors='k', linewidth=0.2)

    # Вычисляем SD1 и SD2
    sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
    sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)

    # Линия идентичности (y = x)
    min_val = min(rr_n.min(), rr_n1.min())
    max_val = max(rr_n.max(), rr_n1.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='y = x')

    # Эллипс SD1/SD2
    mean_rr_n = np.mean(rr_n)
    mean_rr_n1 = np.mean(rr_n1)

    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean_rr_n + sd2 * np.cos(np.pi/4) * np.cos(theta) - sd1 * np.sin(np.pi/4) * np.sin(theta)
    ellipse_y = mean_rr_n1 + sd2 * np.cos(np.pi/4) * np.sin(theta) + sd1 * np.sin(np.pi/4) * np.cos(theta)

    ax4.plot(ellipse_x, ellipse_y, 'r-', linewidth=2, alpha=0.7, label=f'SD1={sd1:.1f}, SD2={sd2:.1f}')

    ax4.set_title("Poincaré Plot (SD1/SD2)")
    ax4.set_xlabel("RR_n (мс)")
    ax4.set_ylabel("RR_n+1 (мс)")
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=9)
    plt.colorbar(sc, ax=ax4, label="Время (мин)")

    # 5. Значения ВСР (включая SD1, SD2)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    features = compute_all_hrv_features(rr_intervals_filtered)
    stats_text = (
        f"Значения ВСР:\n\n"
        f"Временные:\n"
        f"• Mean RR: {features['mean_rr_ms']:.1f} мс\n"
        f"• SDNN: {features['sdnn_ms']:.1f} мс\n"
        f"• RMSSD: {features['rmssd_ms']:.1f} мс\n"
        f"• pNN50: {features['pnn50_percent']:.1f}%\n\n"
        f"Спектральные:\n"
        f"• VLF: {features['vlf_ms2']:.1f} мс²\n"
        f"• LF: {features['lf_ms2']:.1f} мс²\n"
        f"• HF: {features['hf_ms2']:.1f} мс²\n"
        f"• LF/HF: {features['lf_hf_ratio']:.2f}\n\n"
        f"Нелинейные:\n"
        f"• Sample Entropy: {features['sample_entropy']:.3f}\n"
        f"• ApEn: {features.get('approx_entropy', 0.0):.3f}\n"
        f"• FuzzyEn: {features.get('fuzzy_entropy', 0.0):.3f}\n"
        f"• DFA α1: {features['dfa_alpha1']:.3f}\n"
        f"• DFA α2: {features.get('dfa_alpha2', 0.0):.3f}\n"
        f"• MSE mean: {features.get('mse_mean', 0.0):.3f}\n"
        f"• MSE slope: {features.get('mse_slope', 0.0):.3f}\n\n"
        f"Пуанкаре:\n"
        f"• SD1: {sd1:.2f} мс\n"
        f"• SD2: {sd2:.2f} мс\n"
        f"• SD1/SD2: {sd1/sd2:.3f}\n\n"
        f"Индексы:\n"
        f"• ИН: {features['in_index']:.4f}\n"
        f"• ПАПР: {features['papr']:.4f}\n"
        f"• ВПР: {features['vpr']:.4f}"
    )

    ax5.text(0.1, 0.5, stats_text, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=0.8),
             fontsize=9, family='monospace')

    # 6. Информация о фильтрации
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    n_removed = len(rr_intervals_original) - len(rr_intervals_filtered)
    filter_info = (
        f"Параметры анализа:\n\n"
        f"• Исходных RR: {len(rr_intervals_original)}\n"
        f"• После фильтрации: {len(rr_intervals_filtered)}\n"
        f"• Удалено выбросов: {n_removed} ({n_removed/len(rr_intervals_original)*100:.1f}%)\n"
        f"• Длительность: {features['duration_min']:.2f} мин\n\n"
        f"Частотные диапазоны:\n"
        f"• VLF: 0.003-0.04 Гц\n"
        f"• LF: 0.04-0.15 Гц\n"
        f"• HF: 0.15-0.4 Гц"
    )

    ax6.text(0.1, 0.5, filter_info, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='#e7f3ff', alpha=0.8),
             fontsize=9, family='monospace')

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
        if metric not in features or metric not in norms:
            continue
        
        value = features[metric]
        if not pd.notna(value):
            continue
        
        norm = norms[metric]
        desc = norm.get("desc", metric)
        unit = norm.get("unit", "")
        
        interp = get_indicator_interpretation(float(value), metric, norms)
        
        unit_str = f" {unit}" if unit else ""
        value_str = f"{value:.3f}{unit_str}"
        
        #  цвет светофора  в ячейке "Интерпретация" (фон = цвет индикатора, текст — белый).
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
    """Создание HTML-отчета через HTMLReportBuilder.

    Общая сводная панель графиков (plots[test_id]) выводится одной картинкой и
    при клике увеличивается в модальном окне. Размер модального окна
    (зум при клике) увеличен через дополнительный <style> (файл builder'а не
    правится) — см. ZOOM_MODAL_CSS.
    """
    ZOOM_MODAL_CSS = (
        "<style>"
        ".modal { background: rgba(0,0,0,0.94); }"
        ".modal-content { width: 96vw; max-width: 2000px; "
        "box-shadow: 0 0 40px rgba(0,0,0,0.6); }"
        ".modal-content img { width: 100%; height: auto; }"
        ".close-modal { font-size: 2.5rem; }"
        ".block-title { color: #dc3545; border-bottom: 2px solid #dc3545; "
        "padding-bottom: 0.3rem; margin: 1.5rem 0 0.4rem; font-size: 1.2rem; }"
        ".test-subtitle { color: #0d6efd; margin: 0 0 0.8rem; font-size: 1rem; "
        "font-weight: 600; }"
        "</style>"
    )
    builder = HTMLReportBuilder(title="💓 Отчет по анализу вариабельности сердечного ритма")

    # Расширяем зум модального окна при клике (добавляется как сырой HTML).
    builder.add_html(ZOOM_MODAL_CSS, as_section=False)

    for features in features_list:
        test_id = features.get("test_id", "unknown")
        rows_html = build_results_table_vertical(features, norms)
        block = (
            f'<div class="result-block">'
            f'<table class="results-table">'
            f'<thead><tr><th>Показатель</th><th>Значение</th>'
            f'<th>Интерпретация</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
            f'</div>'
        )
        builder.add_html(block, as_section=True, title=test_id)

    for test_id, fig in plots.items():
        builder.add_figure(fig, title="Визуализация")

    builder.save(str(out_path))
    print(f"✅ HTML-отчет сохранен: {out_path}")


def save_xlsx_report(features_list: List[Dict[str, float]], out_path: Path, norms: Dict[str, Dict[str, Any]]):
    """Сохранение результатов в Excel с вертикальной структурой и интерпретацией."""
    display_order = [
        "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_percent",
        "lf_ms2", "hf_ms2", "lf_hf_ratio",
        "sample_entropy", "approx_entropy", "fuzzy_entropy",
        "dfa_alpha1", "dfa_alpha2", "mse_mean", "mse_slope",
        "sd1_ms", "sd2_ms",
        "in_index", "papr", "vpr"
    ]
    
    rows = []
    interp_colors = []  # цвет светофора для каждой строки (под интерпретацию)
    for features in features_list:
        test_id = features.get("test_id", "unknown")
        for metric in display_order:
            if metric not in features or metric not in norms:
                continue
            value = features[metric]
            if not pd.notna(value):
                continue
            norm = norms[metric]
            interp = get_indicator_interpretation(float(value), metric, norms)
            rows.append({
                "Тест": test_id,
                "Показатель": norm.get("desc", metric),
                "Значение": round(float(value), 3),
                "Ед.изм.": norm.get("unit", ""),
                "Интерпретация": interp["interpretation"]
            })
            interp_colors.append(interp["color"])
    
    df = pd.DataFrame(rows)
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Результаты ВСР', index=False)
        worksheet = writer.sheets['Результаты ВСР']
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.column_dimensions[chr(65 + idx)].width = max_len

        # Колонка "Статус" удалена; цвет светофора перенесён на ячейки
        # "Интерпретация" (заливка фона + белый/чёрный текст).
        interp_col = df.columns.get_loc("Интерпретация") + 1  # 1-индексация Excel
        interp_letter = chr(64 + interp_col) if interp_col <= 26 else \
            chr(64 + (interp_col - 1) // 26) + chr(64 + (interp_col - 1) % 26 + 1)
        for row_idx, color in enumerate(interp_colors, start=2):  # строка 1 — заголовок
            cell = worksheet[f"{interp_letter}{row_idx}"]
            font_color = "000000" if color == "#ffc107" else "FFFFFF"
            cell.fill = PatternFill(start_color=color.lstrip("#"),
                                    end_color=color.lstrip("#"),
                                    fill_type="solid")
            cell.font = Font(color=font_color, bold=True)
            
    print(f"✅ XLSX-отчет сохранен: {out_path}")


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
                    except:
                        continue
            except:
                pass
    return data_files


# =============================================================================
# 9. ТОЧКА ВХОДА
# =============================================================================
