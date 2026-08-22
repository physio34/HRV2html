# -*- coding: utf-8 -*-
"""
HTML REPORT BUILDER
Модуль для построения HTML-отчётов с визуализациями
"""
from __future__ import annotations
import base64
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

VERSION = "0.0.6"

class HTMLReportBuilder:
    def __init__(self, title: str = "Отчет анализа данных", fig_dpi: int = 130,
                 version: str = VERSION, source_file: str = ""):
        self.title = title
        self.fig_dpi = fig_dpi
        self.version = version
        self.source_file = Path(source_file).name
        self.sections: List[Dict[str, Any]] = []
        self._html_parts: List[str] = []
        self._fig_counter = 0

    def add_html(self, html_content: str, as_section: bool = True):
        if not html_content:
            return
        cleaned = re.sub(r'[═╔╚║╠╣╦╩╬\-]{2,}', '', html_content)
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned).strip()

        if as_section:
            self.sections.append({"type": "raw_html", "content": cleaned})
        else:
            self._html_parts.append(cleaned)

    def add_text(self, text: str, level: str = "info"):
        self.sections.append({"type": "preview", "content": text, "level": level})

    def add_dataframe(self, df: pd.DataFrame, caption: str = "",
                      note: str = "", expanded: bool = True):
        """Добавляет таблицу. note — текст-комментарий внутри того же блока;
        expanded=False — блок свёрнут по умолчанию."""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df_out = df.copy()
        df_out.columns = df_out.columns.map(str)
        df_out = self._style_validation_table(df_out)

        drop_cols = [c for c in df_out.columns if isinstance(c, str) and c.lower() in ['значимо', 'significant', 'reject']]
        if drop_cols:
            df_out.drop(columns=drop_cols, inplace=True)

        p_cols = [c for c in df_out.columns if 'p-value' in c.lower() or c.lower() in ['p', 'p_value']]
        for col in p_cols:
            def color_p(val):
                try:
                    v = float(val)
                    return f'<span style="color: #ef4444; font-weight: bold;">{v:.4f}</span>' if v < 0.05 else f'<span style="color: #10b981;">{v:.4f}</span>'
                except:
                    return val
            df_out[col] = df_out[col].apply(color_p)

        for col in df_out.select_dtypes(include=['float', 'float64', 'float32']).columns:
            if col not in p_cols:
                df_out[col] = df_out[col].round(2)

        self.sections.append({"type": "dataframe", "content": df_out,
                              "caption": caption, "note": note, "expanded": expanded})

    def add_figure(self, fig: plt.Figure, title: str = "", caption: str = "", group_title: Optional[str] = None):
        """Добавляет фигуру. При указании group_title последовательные фигуры с одинаковым group_title группируются."""
        self._fig_counter += 1
        self.sections.append({
            "type": "figure",
            "content": fig,
            "title": title,
            "caption": caption,
            "group_title": group_title
        })

    def add_summary_cards(self, cards: List[Dict[str, Any]], title: str = "Сводка",
                          note: str = ""):
        self.sections.append({"type": "summary", "content": cards,
                              "group_title": title, "note": note})

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.fig_dpi, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def save(self, filepath: str) -> str:
        Path(filepath).write_text(self._generate_html(), encoding="utf-8")
        return filepath

    def _style_validation_table(self, df: pd.DataFrame) -> pd.DataFrame:
        styled = df.copy()

        def color_pvalue(val):
            try:
                p = float(val)
                color = "#10b981" if p < 0.05 else "#ef4444"
                return f'<span style="color:{color}; font-weight:bold;">{p:.4f}</span>'
            except Exception:
                return val

        def color_bool(val):
            if pd.isna(val):
                return val
            is_true = val is True or str(val).strip().upper() == "TRUE"
            color = "#10b981" if is_true else "#ef4444"
            return f'<span style="color:{color}; font-weight:bold;">{val}</span>'

        for col in styled.columns:
            col_lower = str(col).lower()
            if "p-value" in col_lower or col_lower in ["p", "p_value", "pvalue", "p-adj"]:
                styled[col] = styled[col].apply(color_pvalue)
            if any(key in col_lower for key in ["значимо", "significant", "reject", "true/false"]):
                styled[col] = styled[col].apply(color_bool)
        return styled

    def _generate_html(self) -> str:
        # Группировка фигур по group_title
        grouped_sections = []
        i = 0
        while i < len(self.sections):
            sec = self.sections[i]
            if sec["type"] == "figure" and sec.get("group_title") is not None:
                group_title = sec["group_title"]
                fig_group = []
                # Собираем все последовательные фигуры с таким же group_title
                while i < len(self.sections) and self.sections[i]["type"] == "figure" and self.sections[i].get("group_title") == group_title:
                    fig_group.append(self.sections[i])
                    i += 1
                grouped_sections.append({"type": "figure_group", "content": fig_group, "group_title": group_title})
            else:
                grouped_sections.append(sec)
                i += 1

        sections_html = []
        for section in grouped_sections:
            s_type = section["type"]
            rendered = " "
            title = " "
            icon = "📊 "

            if s_type == "raw_html":
                rendered = section["content"]
                title = " "

            elif s_type == "preview":
                icon = "📄 "
                title = self.source_file
                rendered = f'<div class="preview-card">{section["content"]}</div>'

            elif s_type == "dataframe":
                caption = section.get("caption", " ")
                icon = "📋 "
                title = caption
                table_html = section["content"].to_html(classes="dataframe", index=True, escape=False)
                note_html = ""
                note = str(section.get("note", "")).strip()
                if note:
                    note_html = f'<div class="block-note">{note}</div>'
                rendered = f'{note_html}<div class="table-container">{table_html}</div>'

            elif s_type == "figure_group":
                figs = section["content"]
                icon = "📈 "
                title = str(section.get("group_title", "")).strip()
                is_grid = len(figs) >= 3
                grid_class = "plots-grid" if is_grid else "plots-stack"

                figs_html = []
                for fig_sec in figs:
                    img_b64 = self._fig_to_base64(fig_sec["content"])
                    fig_title = str(fig_sec.get("title", "")).strip()
                    fig_caption = str(fig_sec.get("caption", "")).strip()

                    # Основной текст для alt и заголовка
                    alt_text = fig_title if fig_title else fig_caption

                    # ИЗБЕГАЕМ ДУБЛИРОВАНИЯ: выводим подпись под картинкой только если она отличается от заголовка
                    caption_text = fig_caption if (fig_caption and fig_caption != alt_text) else ""

                    figs_html.append(
                        f'<div class="plot-item">'
                        f'<img src="data:image/png;base64,{img_b64}" alt="{alt_text}" '
                        f'class="plot-image zoomable" onclick="openModal(this)" title="Нажмите для увеличения">'
                        f'{"<p class=\"plot-caption\">" + caption_text + "</p>" if caption_text else ""}'
                        f'</div>'
                    )
                rendered = f'<div class="{grid_class}">{"  ".join(figs_html)}</div>'

            elif s_type == "figure":  # Одиночная фигура (без группы)
                img_b64 = self._fig_to_base64(section["content"])
                fig_title = str(section.get("title", "")).strip()
                fig_caption = str(section.get("caption", "")).strip()
                icon = "📈 "

                section_title = fig_title if fig_title else fig_caption
                title = section_title

                # ИЗБЕГАЕМ ДУБЛИРОВАНИЯ: выводим подпись только если она отличается от section_title
                caption_text = fig_caption if (fig_caption and fig_caption != section_title) else ""

                rendered = (
                    f'<div class="plot-item">'
                    f'<img src="data:image/png;base64,{img_b64}" alt="{section_title}" '
                    f'class="plot-image zoomable" onclick="openModal(this)" title="Нажмите для увеличения">'
                    f'{"<p class=\"plot-caption\">" + caption_text + "</p>" if caption_text else ""}'
                    f'</div>'
                )

            elif s_type == "summary":
                icon = "📊 "
                title = section.get("group_title", "Сводка")
                cards = " ".join(
                    f'<div class="card"><div class="card-value">{c["value"]}</div><div class="card-label">{c["label"]}</div></div>'
                    for c in section["content"]
                )
                note_html = ""
                note = str(section.get("note", "")).strip()
                if note:
                    note_html = f'<div class="block-note">{note}</div>'
                rendered = f'{note_html}<div class="card-grid">{cards}</div>'

            if title or rendered:
                block_open = " open" if section.get("expanded", True) else ""
                sections_html.append(
                    f'<details class="block-layer"{block_open}>'
                    f'<summary>{icon}{title}</summary>'
                    f'<div class="block-content">{rendered}</div>'
                    f'</details>'
                )

        now = pd.Timestamp.now().strftime("%d.%m.%Y %H:%M:%S")
        return f"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
<style>{self._get_css()}</style>
</head>
<body>
<div class="container">
<div class="header"><h1>📊 {self.title}</h1></div>
<div class="content">
{"\n".join(sections_html)}
{" ".join(self._html_parts)}
</div>
<div class="footer">
<div>Автоматический анализ данных | Версия: {self.version}</div>
<div style="margin-top: 8px; opacity: 0.7;">📅 Сгенерировано: {now}</div>
</div>
</div>
<div id="imageModal" class="modal" onclick="closeModal()">
    <span class="close">&times;</span>
    <img class="modal-content" id="modalImg">
</div>
<script>
    function openModal(img) {{ document.getElementById("imageModal").style.display = "flex"; document.getElementById("modalImg").src = img.src; }}
    function closeModal() {{ document.getElementById("imageModal").style.display = "none"; }}
    document.addEventListener('keydown', e => e.key === "Escape" && closeModal());
</script>
</body>
</html>
"""

    def _get_css(self) -> str:
        return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui; background: #0f172a; color: #e2e8f0; padding: 20px; }
.container { max-width: 1400px; margin: 0 auto; }
.header { background: #1e293b; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #334155; text-align: center; }
.header h1 { color: #3b82f6; font-size: 2rem; }
.block-layer { background: #1e293b; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #334155; overflow: hidden; }
.block-layer summary { padding: 1rem 1.5rem; cursor: pointer; font-weight: 600; color: #60a5fa; font-size: 1.2rem; border-bottom: 1px solid #334155; background: #0f172a; transition: background 0.2s; list-style: none; display: flex; align-items: center; gap: 10px; }
.block-layer summary:hover { background: #1e293b; }
.block-layer summary::after { content: "▼"; margin-left: auto; transition: transform 0.3s; }
.block-layer[open] summary::after { transform: rotate(180deg); }
.block-content { padding: 1.5rem; }
.table-container { overflow-x: auto; }
.block-note { background: #0f172a; border: 1px solid #334155; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem; color: #94a3b8; font-size: 0.9em; line-height: 1.55; white-space: pre-line; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }
th { background: #0f172a; color: #60a5fa; }
.preview-card { background: linear-gradient(135deg, #1e3a8a, #0f172a); border: 1px solid #3b82f6; border-radius: 8px; padding: 1.2rem; margin-bottom: 0.5rem; color: #e2e8f0; font-size: 0.95em; line-height: 1.5; }
.plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
.plots-stack { display: flex; flex-direction: column; gap: 1.5rem; }
.plot-item { background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid #334155; padding: 10px; }
.plot-image { width: 100%; height: auto; display: block; cursor: zoom-in; transition: transform 0.2s; border-radius: 6px; }
.plot-image:hover { transform: scale(1.01); }
.plot-caption { text-align: center; margin-top: 8px; font-size: 0.85em; color: #94a3b8; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }
.card { background: #0f172a; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155; }
.card-value { font-size: 1.8em; font-weight: bold; color: #60a5fa; }
.card-label { color: #94a3b8; font-size: 0.85em; margin-top: 5px; }
.footer { text-align: center; padding: 1.5rem; color: #64748b; border-top: 1px solid #334155; margin-top: 2rem; font-size: 0.9em; }
.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.95); justify-content: center; align-items: center; cursor: zoom-out; }
.modal-content { max-width: 90%; max-height: 90%; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
.close { position: absolute; top: 20px; right: 30px; color: #fff; font-size: 40px; font-weight: bold; cursor: pointer; }
"""