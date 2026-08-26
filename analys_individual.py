"""💓 Универсальный анализатор вариабельности сердечного ритма (ВСР).

Тонкая обёртка поверх общего блока анализа (modules.hrvanalysis) и
генератора отчётов (modules.html_report_builder).
XLSX-отчёт этим скриптом НЕ формируется (только HTML).
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
import matplotlib.pyplot as plt  # noqa: F401  (нужен для построения графиков в блоке анализа)

from modules.html_report_builder import HTMLReportBuilder  # noqa: F401 (переэкспорт для тестов)
from modules.hrvanalysis import (
    compute_all_hrv_features,
    compute_sample_entropy,
    compute_entropy_features,
    remove_outliers_moving_average,
    load_rr_intervals_from_file,
    preprocess_rr_intervals,
    plot_hrv_analysis,
    build_results_table_vertical,
    save_html_report,
    save_xlsx_report,
    load_norms_config,
    discover_data_files,
    DEFAULT_NORMS,
)


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="💓 Универсальный анализатор вариабельности сердечного ритма (ВСР)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                                # Интерактивный режим
  %(prog)s --dir /path/to/data            # Обработка директории
  %(prog)s --file data.csv                # Обработка конкретного файла
  %(prog)s --dir /data --all              # Обработать все файлы без подтверждения
  %(prog)s --config my_norms.json         # Использовать свой конфиг нормативов
        """
    )
    parser.add_argument('--dir', type=str, help='Путь к директории с файлами данных')
    parser.add_argument('--file', type=str, help='Путь к конкретному файлу')
    parser.add_argument('--all', action='store_true', help='Обработать все файлы без подтверждения')
    parser.add_argument('--config', type=str, help='Путь к JSON-файлу нормативов')
    return parser.parse_args()


def main():
    # Принудительно UTF-8 вывод, чтобы эмодзи/кириллица не падали в консолях Windows (cp1252).
    for _s in (sys.stdout, sys.stderr):
        try:
            if _s is not None and hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("💓 УНИВЕРСАЛЬНЫЙ АНАЛИЗАТОР ВАРИАБЕЛЬНОСТИ СЕРДЕЧНОГО РИТМА")
    print("=" * 60)

    args = parse_arguments()

    config_path = Path(args.config) if args.config else None
    norms = load_norms_config(config_path)

    if args.file:
        file_path = Path(args.file).resolve()
        if not file_path.is_file():
            print(f"❌ Файл не найден: {file_path}")
            return
        print(f"\n📄 Режим: Обработка конкретного файла")
        print(f"   Файл: {file_path.name}")
        selected = [{"id": file_path.stem, "path": file_path}]
        root_dir = file_path.parent

    elif args.dir:
        root_dir = Path(args.dir).resolve()
        if not root_dir.is_dir():
            print(f"❌ Директория не найдена: {root_dir}")
            return
        print(f"\n📁 Режим: Обработка директории")
        print(f"   Путь: {root_dir}")
        print("🔍 Поиск файлов...")
        csv_files = discover_data_files(root_dir)
        if not csv_files:
            print("❌ Не найдено подходящих CSV/TXT файлов.")
            return
        if args.all:
            selected = csv_files
        else:
            print(f"\nНайдено: {len(csv_files)} файлов.")
            for i, item in enumerate(csv_files, 1):
                print(f"  {i}. {item['id']}")
            choice = input("\n📥 Номера через запятую или 'all': ").strip()
            if choice.lower() == 'all':
                selected = csv_files
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(",")]
                    selected = [csv_files[i] for i in indices if 0 <= i < len(csv_files)]
                except ValueError:
                    print("⚠️ Ошибка ввода. Обработаны все файлы.")
                    selected = csv_files
    else:
        dir_input = input("\n📁 Путь к директории (Enter = текущая): ").strip() or "."
        root_dir = Path(dir_input).resolve()
        if not root_dir.is_dir():
            print("❌ Директория не найдена.")
            return
        print("🔍 Поиск файлов...")
        csv_files = discover_data_files(root_dir)
        if not csv_files:
            print("❌ Не найдено подходящих CSV/TXT файлов.")
            return
        print(f"\nНайдено: {len(csv_files)} файлов.")
        for i, item in enumerate(csv_files, 1):
            print(f"  {i}. {item['id']}")
        choice = input("\n📥 Номера через запятую или 'all': ").strip()
        if choice.lower() == 'all':
            selected = csv_files
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [csv_files[i] for i in indices if 0 <= i < len(csv_files)]
            except ValueError:
                print("⚠️ Ошибка ввода. Обработаны все файлы.")
                selected = csv_files

    features_list = []
    plots_dict = {}

    for idx, item in enumerate(selected, 1):
        print(f"\n🔄 [{idx}/{len(selected)}] {item['id']}...")
        try:
            rr_intervals_original = load_rr_intervals_from_file(item["path"])
            rr_intervals_original = preprocess_rr_intervals(rr_intervals_original, min_duration_min=2.0)

            # Сохраняем исходные данные для визуализации
            rr_intervals_for_plot = rr_intervals_original.copy()

            # Фильтрация выбросов методом скользящего среднего
            original_count = len(rr_intervals_original)
            rr_intervals_filtered, n_removed = remove_outliers_moving_average(rr_intervals_original, window_size=5, threshold=3.0)
            print(f"   🔍 Фильтрация: удалено {n_removed} выбросов из {original_count} ({n_removed/original_count*100:.1f}%)")

            if len(rr_intervals_filtered) < 50:
                raise ValueError(f"Слишком мало RR-интервалов после фильтрации ({len(rr_intervals_filtered)})")

            feats = compute_all_hrv_features(rr_intervals_filtered)

            # Вычисляем SD1 и SD2 для добавления в результаты
            rr_n = rr_intervals_filtered[:-1]
            rr_n1 = rr_intervals_filtered[1:]
            sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
            sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)
            feats["sd1_ms"] = round(sd1, 2)
            feats["sd2_ms"] = round(sd2, 2)

            feats["test_id"] = item["id"]
            feats["n_removed_outliers"] = n_removed
            features_list.append(feats)

            fig = plot_hrv_analysis(rr_intervals_for_plot, rr_intervals_filtered, title=item["id"])
            plots_dict[item["id"]] = fig

            print(f"   ✅ OK | Mean RR: {feats['mean_rr_ms']:.1f} мс | SDNN: {feats['sdnn_ms']:.1f} мс | SD1: {sd1:.1f} мс | SD2: {sd2:.1f} мс")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            traceback.print_exc()

    if features_list:
        if len(selected) == 1:
            base_name = selected[0]["path"].stem
            report_name = f"hrv_report_{base_name}"
        else:
            base_name = root_dir.name
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"hrv_report_{base_name}_{timestamp}"

        save_html_report(features_list, plots_dict, root_dir / f"{report_name}.html", norms)
        # XLSX-отчёт этим скриптом не формируется (только HTML).
        print("\n✨ Обработка успешно завершена!")
    else:
        print("\n⚠️ Не удалось обработать ни один файл.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹ Прерывание пользователем.")
        sys.exit(1)
