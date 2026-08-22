"""💓 Групповой анализ ВСР с расширенным нелинейным блоком (энтропии).

Обходит все .txt файлы в выбранном каталоге (по умолчанию — текущем),
использует тот же блок анализа, что и analys_html_individual_entropy.py
(modules.hrvanalysis), и формирует ОДИН общий XLSX-отчёт.
"""
import os
import sys
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from modules.hrvanalysis import (
    compute_all_hrv_features,
    remove_outliers_moving_average,
    load_rr_intervals_from_file,
    preprocess_rr_intervals,
    compute_spectral_peak_features,
)

# Базовые («уже имеющиеся») колонки — метрики блока анализа entropy-модуля.
BASE_COLUMNS = [
    "mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_percent",
    "lf_ms2", "hf_ms2", "lf_hf_ratio", "lf_norm", "hf_norm",
    "peak_vlf_hz", "peak_lf_hz", "peak_hf_hz",
    "sample_entropy", "approx_entropy", "fuzzy_entropy",
    "dfa_alpha1", "dfa_alpha2", "mse_mean", "mse_slope",
    "sd1_ms", "sd2_ms",
    "in_index", "papr", "vpr",
]
# Дополнительные колонки согласно ТЗ (fraction_lf_nu/fraction_hf_nu исключены).
EXTRA_COLUMNS = [
    "sd2_sd1_ratio",
]
REPORT_COLUMNS = ["filename"] + BASE_COLUMNS + EXTRA_COLUMNS


def find_txt_files(directory: Path) -> List[Path]:
    """Все .txt/.TXT файлы в каталоге (рекурсивно)."""
    files = set(directory.rglob("*.txt")) | set(directory.rglob("*.TXT"))
    return sorted(files)


def analyze_file(fpath: Path) -> Dict[str, Any]:
    """Полный анализ одного файла; возвращает строку для общего отчёта."""
    rr_original = load_rr_intervals_from_file(fpath)
    rr_original = preprocess_rr_intervals(rr_original, min_duration_min=2.0)

    rr_filtered, n_removed = remove_outliers_moving_average(
        rr_original, window_size=5, threshold=3.0)
    if len(rr_filtered) < 50:
        raise ValueError(f"Слишком мало RR после фильтрации ({len(rr_filtered)})")

    feats = compute_all_hrv_features(rr_filtered)

    # Poincaré SD1/SD2 и их отношение.
    rr_n = rr_filtered[:-1]
    rr_n1 = rr_filtered[1:]
    sd1 = float(np.std(rr_n1 - rr_n) / np.sqrt(2))
    sd2 = float(np.std(rr_n1 + rr_n) / np.sqrt(2))
    feats["sd1_ms"] = round(sd1, 2)
    feats["sd2_ms"] = round(sd2, 2)
    feats["sd2_sd1_ratio"] = round(sd2 / sd1, 3) if sd1 > 0 else np.nan

    # Спектральные пики и нормированные доли.
    peaks = compute_spectral_peak_features(rr_filtered)
    feats.update(peaks)

    row = {"filename": fpath.name}
    for col in BASE_COLUMNS + EXTRA_COLUMNS:
        val = feats.get(col, np.nan)
        row[col] = val if pd.notna(val) else np.nan
    return row


def save_group_xlsx(rows: List[Dict[str, Any]], out_path: Path):
    """Сохраняет общий отчёт: лист HRV_Results + лист Statistics."""
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    for col in df.columns:
        if col != "filename":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_disp = df.copy()
        for col in df_disp.select_dtypes(include=["float64", "float32"]).columns:
            df_disp[col] = df_disp[col].round(3)
        df_disp.to_excel(writer, sheet_name="HRV_Results", index=False, na_rep="")

        if len(df) > 1:
            numeric = df.select_dtypes(include=["float64", "float32"]).round(3)
            numeric.describe().to_excel(writer, sheet_name="Statistics", na_rep="")

        ws = writer.sheets["HRV_Results"]
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    max_len = max(max_len, len(str(cell.value)))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 50)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="💓 Групповой анализ ВСР (энтропийный блок) → общий XLSX-отчёт")
    parser.add_argument('--dir', type=str, default='.',
                        help='Каталог с .txt файлами (по умолчанию — текущий)')
    parser.add_argument('--output', type=str, default='hrv_group_report.xlsx',
                        help='Путь к итоговому XLSX')
    return parser.parse_args()


def main():
    print("=" * 60)
    print("💓 ГРУППОВОЙ АНАЛИЗ ВСР (ЭНТРОПИЙНЫЙ БЛОК)")
    print("=" * 60)

    args = parse_arguments()
    root_dir = Path(args.dir).resolve()
    if not root_dir.is_dir():
        print(f"❌ Директория не найдена: {root_dir}")
        return

    files = find_txt_files(root_dir)
    if not files:
        print(f"❌ Не найдено .txt файлов в: {root_dir}")
        return
    print(f"🔍 Найдено .txt файлов: {len(files)}")

    rows = []
    skipped = []
    for i, fpath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fpath.name} ...")
        try:
            rows.append(analyze_file(fpath))
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            traceback.print_exc()
            skipped.append((fpath.name, str(e)))

    if not rows:
        print("❌ Нет данных для сохранения.")
        return

    out_path = Path(args.output)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".xlsx")
    save_group_xlsx(rows, out_path)

    print("\n" + "=" * 60)
    print("КРАТКАЯ СВОДКА")
    print("=" * 60)
    print(f"✅ Успешно: {len(rows)} из {len(files)}")
    if skipped:
        for name, reason in skipped:
            print(f"  ⚠️ {name}: {reason}")
    print(f"📊 XLSX сохранён: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹ Прерывание пользователем.")
        sys.exit(1)
