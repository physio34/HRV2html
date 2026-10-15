"""💓 Универсальный анализатор вариабельности сердечного ритма (ВСР).

Тонкая обёртка поверх общего блока анализа (modules.hrvanalysis) и
генератора отчётов (modules.html_report_builder).
XLSX-отчёт этим скриптом НЕ формируется (только HTML).
"""
import os
import sys
import argparse
import traceback
import webbrowser
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


# ── ANSI-цвета для вывода в консоль ──────────────────────────────────────────
_COLORS_ENABLED = False

class C:
    """ANSI escape-коды для цветного вывода."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    GRAY    = "\033[90m"

def _enable_color():
    """Включение ANSI-цветов (Windows 10+ поддерживает через virtual terminal)."""
    global _COLORS_ENABLED
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
    _COLORS_ENABLED = True

def _c(text: str, color: str) -> str:
    """Обернуть текст в ANSI-цвет, если цвет включён."""
    if not _COLORS_ENABLED:
        return str(text)
    return f"{color}{text}{C.RESET}"


def _sort_files_by_date(files: List[Dict]) -> List[Dict]:
    """Сортировка файлов по дате модификации (новые первыми)."""
    def _mtime(item: Dict) -> float:
        try:
            return item["path"].stat().st_mtime
        except OSError:
            return 0.0
    return sorted(files, key=_mtime, reverse=True)


def _print_file_list(files: List[Dict]) -> None:
    """Красивый вывод списка файлов с датами."""
    print(f"\n{_c(f'Найдено: {len(files)} файлов', C.BOLD)}")
    for i, item in enumerate(files, 1):
        try:
            mtime = item["path"].stat().st_mtime
            date_str = pd.Timestamp.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            date_colored = _c(f"[{date_str}]", C.GRAY)
        except OSError:
            date_colored = ""
        print(f"  {_c(i, C.CYAN)}. {item['id']}  {date_colored}")


def _prompt_file_selection(files: List[Dict]) -> List[Dict]:
    """Запрос выбора файлов у пользователя."""
    choice = input(f"\n{_c('📥', C.BLUE)} Номера через запятую или 'all': ").strip()
    if choice.lower() == 'all':
        return files
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [files[i] for i in indices if 0 <= i < len(files)]
        if not selected:
            print(_c("⚠️ Нет валидных номеров. Обработаны все файлы.", C.YELLOW))
            return files
        return selected
    except ValueError:
        print(_c("⚠️ Ошибка ввода. Обработаны все файлы.", C.YELLOW))
        return files


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
    parser.add_argument('--no-open', action='store_true', help='Не открывать HTML-отчёт после завершения')
    return parser.parse_args()


def main():
    # Принудительно UTF-8 вывод, чтобы эмодзи/кириллица не падали в консолях Windows (cp1252).
    for _s in (sys.stdout, sys.stderr):
        try:
            if _s is not None and hasattr(_s, "reconfigure"):
                _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    _enable_color()

    print(_c("=" * 60, C.CYAN))
    print(_c("💓 УНИВЕРСАЛЬНЫЙ АНАЛИЗАТОР ВАРИАБЕЛЬНОСТИ СЕРДЕЧНОГО РИТМА", C.BOLD))
    print(_c("=" * 60, C.CYAN))

    args = parse_arguments()

    config_path = Path(args.config) if args.config else None
    norms = load_norms_config(config_path)

    if args.file:
        file_path = Path(args.file).resolve()
        if not file_path.is_file():
            print(_c(f"❌ Файл не найден: {file_path}", C.RED))
            return
        print(f"\n{_c('📄', C.BLUE)} Режим: Обработка конкретного файла")
        print(f"   Файл: {_c(file_path.name, C.BOLD)}")
        selected = [{"id": file_path.stem, "path": file_path}]
        root_dir = file_path.parent

    elif args.dir:
        root_dir = Path(args.dir).resolve()
        if not root_dir.is_dir():
            print(_c(f"❌ Директория не найдена: {root_dir}", C.RED))
            return
        print(f"\n{_c('📁', C.BLUE)} Режим: Обработка директории")
        print(f"   Путь: {_c(str(root_dir), C.BOLD)}")
        print(f"{_c('🔍', C.BLUE)} Поиск файлов...")
        csv_files = discover_data_files(root_dir)
        if not csv_files:
            print(_c("❌ Не найдено подходящих CSV/TXT файлов.", C.RED))
            return
        csv_files = _sort_files_by_date(csv_files)
        if args.all:
            selected = csv_files
        else:
            _print_file_list(csv_files)
            selected = _prompt_file_selection(csv_files)
    else:
        dir_input = input(f"\n{_c('📁', C.BLUE)} Путь к директории (Enter = текущая): ").strip() or "."
        root_dir = Path(dir_input).resolve()
        if not root_dir.is_dir():
            print(_c("❌ Директория не найдена.", C.RED))
            return
        print(f"{_c('🔍', C.BLUE)} Поиск файлов...")
        csv_files = discover_data_files(root_dir)
        if not csv_files:
            print(_c("❌ Не найдено подходящих CSV/TXT файлов.", C.RED))
            return
        csv_files = _sort_files_by_date(csv_files)
        _print_file_list(csv_files)
        selected = _prompt_file_selection(csv_files)

    features_list = []
    plots_dict = {}

    for idx, item in enumerate(selected, 1):
        print(f"\n{_c('🔄', C.BLUE)} [{idx}/{len(selected)}] {_c(item['id'], C.BOLD)}...")
        try:
            rr_intervals_original, time_column = load_rr_intervals_from_file(item["path"])
            rr_intervals_original = preprocess_rr_intervals(rr_intervals_original, min_duration_min=2.0)

            # Сохраняем исходные данные для визуализации
            rr_intervals_for_plot = rr_intervals_original.copy()

            # Фильтрация выбросов методом скользящего среднего
            original_count = len(rr_intervals_original)
            rr_intervals_filtered, n_removed, time_column_filtered = remove_outliers_moving_average(
                rr_intervals_original, window_size=5, threshold=3.0, time_column=time_column)
            print(f"   {_c('🔍', C.BLUE)} Фильтрация: удалено {_c(n_removed, C.YELLOW)} выбросов из {original_count} ({n_removed/original_count*100:.1f}%)")

            if len(rr_intervals_filtered) < 50:
                raise ValueError(f"Слишком мало RR-интервалов после фильтрации ({len(rr_intervals_filtered)})")

            feats = compute_all_hrv_features(rr_intervals_filtered, time_column=time_column_filtered)

            # SD1/SD2 уже посчитаны в compute_all_hrv_features (ddof=1); берём их
            rr_n = rr_intervals_filtered[:-1]
            rr_n1 = rr_intervals_filtered[1:]
            sd1 = float(feats.get("sd1_ms", float("nan")))
            sd2 = float(feats.get("sd2_ms", float("nan")))

            feats["test_id"] = item["id"]
            feats["n_removed_outliers"] = int(n_removed)
            features_list.append(feats)

            fig = plot_hrv_analysis(rr_intervals_for_plot, rr_intervals_filtered,
                                    title=item["id"], time_column=time_column_filtered,
                                    n_removed_outliers=int(n_removed))
            plots_dict[item["id"]] = fig

            print(f"   {_c('✅', C.GREEN)} OK | Mean RR: {feats['mean_rr_ms']:.1f} мс | SDNN: {feats['sdnn_ms']:.1f} мс | SD1: {sd1:.1f} мс | SD2: {sd2:.1f} мс")
        except Exception as e:
            print(f"   {_c('❌', C.RED)} Ошибка: {_c(e, C.RED)}")
            traceback.print_exc()

    if features_list:
        if len(selected) == 1:
            base_name = selected[0]["path"].stem
            report_name = f"hrv_report_{base_name}"
        else:
            base_name = root_dir.name
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"hrv_report_{base_name}_{timestamp}"

        report_path = root_dir / f"{report_name}.html"
        save_html_report(features_list, plots_dict, report_path, norms)
        print(f"\n{_c('✨', C.GREEN)} Обработка успешно завершена!")
        if not args.no_open:
            print(f"{_c('🌐', C.BLUE)} Открытие отчёта...")
            webbrowser.open(str(report_path))
    else:
        print(f"\n{_c('⚠️', C.YELLOW)} Не удалось обработать ни один файл.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹ Прерывание пользователем.")
        sys.exit(1)
