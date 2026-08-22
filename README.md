# adHRV (adHRV2html) — Heart Rate Variability (HRV) Analyzer

A set of scripts for advanced HRV analysis with emphasis on **nonlinear
(entropy) metrics**: Sample Entropy (KDTree), Approximate Entropy, Fuzzy Entropy,
Multiscale Entropy and detrended fluctuation analysis (DFA α1/α2).

Analysis-core version: `modules/hrvanalysis.__version__`.

## Structure

```
adHRV2html/
├── modules/
│   ├── __init__.py
│   ├── hrvanalysis.py          # shared analysis core (metrics, plots, reports), __version__
│   ├── html_report_builder.py  # HTML report generator (clickable plots)
│   └── hrv_norms.json          # normative config ("traffic light")
├── analys_individual.py        # ONE file/dir  -> HTML report
└── analys_group.py             # ALL .txt in dir -> single XLSX report
```

Imports come from the `modules` package, so the scripts are run from the root
of `adHRV2html/` (it is added to `sys.path` automatically on launch and in tests).

## Dependencies

```bash
pip install numpy scipy pandas matplotlib openpyxl antropy
```

- `antropy` is used as a fallback (reference) implementation for Sample/Fuzzy/
  Multiscale Entropy; the primary Sample Entropy is a custom `scipy.spatial.cKDTree`
  implementation.
- `neurokit2` is removed (see below).

## Unified spectral engine

`analys_individual.py`, `analys_group.py` and `analys_with_neurokit2.py` produce
**identical** spectral quantities (VLF/LF/HF/TP, LF/HF, normalized units LF nu /
HF nu) because all of them use `modules.hrvanalysis.compute_frequency_domain_features`
(Welch method, resampling to a uniform 4 Hz grid, **linear** interpolation,
trapezoidal integration). Previously `analys_with_neurokit2.py` computed the
spectrum via `neurokit2.hrv_frequency` (values in s², inconsistent with the PSD
plot) — this has been fixed.

## Usage

**Individual report (HTML only):**
```bash
python analys_individual.py --file data.txt
python analys_individual.py --dir ./data --all
```
This script does not produce an XLSX (only HTML with interactive plots).

**Group report (single XLSX, no HTML):**
```bash
python analys_group.py --dir . --output hrv_group_report.xlsx
```
Recursively scans all `*.txt` in the chosen directory (current by default), uses
the same analysis core as the individual script, and writes one summary table.
Columns: the base metrics of the analysis core (time-domain, spectral, entropy,
nonlinear, Poincaré, Baevsky indices) **plus** `sd2_sd1_ratio`, `peak_vlf_hz`,
`peak_lf_hz`, `peak_hf_hz`, `fraction_lf_nu`, `fraction_hf_nu`.

---

## Methodological recommendations for entropy metrics

Entropy metrics characterize the **complexity/irregularity** of the RR signal.
Their interpretation and reproducibility depend strongly on data preparation and
parameter choice — below is the rule set followed by the code in
`modules/hrvanalysis.py`.

### 1. Preparation and preprocessing
- **Artifact removal** is mandatory: outliers (>3σ of the moving average, window 5)
  distort both spectrum and entropy. At least ≥ 50 RR intervals must remain after
  filtering (otherwise the script errors; entropy needs more — see §2).
- **Spectrum resampling** goes to a uniform 4 Hz grid (**linear** `np.interp`
  interpolation). Entropy metrics are computed on the **original RR series** (not
  the resampled one) to avoid injecting artificial correlation.
- The signal is **centered** (mean subtracted) before the Welch spectrum.

### 2. Recording-length requirements
| Metric | Min RR | Comment |
|--------|--------|---------|
| Sample / Approximate / Fuzzy Entropy | ~200–250 | for m=2; shorter is unstable |
| Multiscale Entropy | ≥ 500–1000 | needs scales up to 20 |
| DFA α1 / α2 | ≥ 100–150 | α1 — short scales (4–16), α2 — long (17–64) |
| VLF spectrum | duration ≥ 5 min | otherwise frequency resolution (df≈4/256≈0.016 Hz) underestimates VLF |

If the recording is too short, the metric returns `NaN`/0.0 and is excluded from
the table (the "Status" column is absent; the "traffic light" color is moved to
the "Interpretation" cell).

### 3. Entropy parameters (m, r)
- **m = 2** (embedding dimension) — HRV standard.
- **r** (tolerance):
  - Sample Entropy / Approximate Entropy: `r = 0.2` × signal SD (Richman–Moorman).
  - Fuzzy Entropy: `r = 0.15`, exponent `n = 2`.
- Entropies are **sensitive to r**: compare groups only with **identical m and r**,
  otherwise differences may be a parameter artifact rather than physiology.

### 4. Interpretation
- **High entropy** (SampEn, FuzzyEn, MSE) = high complexity/adaptability, normally
  higher in healthy subjects. **Decrease** — in pathology, stress, aging, diabetes, HF.
- **MSE mean / slope**: mean across scales and the slope. Flat/declining slope —
  loss of complexity at larger scales.
- **DFA α1** ≈ 1.0 (anti-persistent noise-like process), **α2** > α1 usually;
  deviation of α1 from ~1 indicates changed correlation structure.
- Entropies are **relative**: absolute values depend on implementation, so use
  **within-cohort/within-sample comparisons** first, not absolute thresholds.

### 5. Spectral fractions (nu) vs LF/HF ratio
- `Fraction LF (nu)` / `Fraction HF (nu)` — **normalized units**:
  `LF/(LF+HF)·100`, `HF/(LF+HF)·100`. They reflect the **balance** of sympathetic-
  vagal regulation, cleansed of total power level.
- `LF/HF ratio` — simple ratio of absolute powers. Sensitive to total power and
  the VLF component.
- Use **nu** for between-group balance comparisons, **ratio** for a general
  vagosympathetic tone judgment.
- `Peak VLF/LF/HF (Hz)` — PSD peak frequencies in the respective bands; useful to
  assess central-frequency shifts (e.g., respiratory modulation).

### 6. Reproducibility
- For comparability keep a **single pipeline**: one filtering method, one resampling
  fs (4 Hz), one m/r, identical minimum duration.
- Do not mix entropy implementations (here — custom KDTree + optional `antropy`
  fallback); in publications specify the library/version.
- VLF power on short recordings is systematically **underestimated** due to
  frequency resolution — interpret it cautiously and only for long (≥5 min) records.

### 7. Normative configuration
- Normatives ("traffic light") are stored in `modules/hrv_norms.json`. If the file
  is absent, built-in values are used. The `--config` key lets you supply your own
  set (to recompute interpretations/colors).

### 8. Nonlinear analysis performance
Benchmarks on 3040 RR after filtering; see `WHATS.new`:

| Script | Nonlinear block | Time |
|---|---|---|
| `analys_individual.py` | `compute_entropy_features` (SampEn, DFA α1/α2, ApEn, FuzzyEn, MSE) | **≈ 3.5 s** |
| `analys_with_neurokit2.py` | `nk.hrv_nonlinear` (SD1, SD2, SampEn, DFA α1) | **≈ 18.6 s** |

`analys_with_neurokit2.py` is ~5× slower even though it computes *fewer* nonlinear
metrics (no ApEn/FuzzyEn/MSE) — the bottleneck is the internal SampEn/DFA
implementation in `neurokit2`. For maximum speed use `analys_individual.py` /
`analys_group.py`.
