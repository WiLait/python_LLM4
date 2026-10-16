import os
import csv
import json
import textwrap
import pandas as pd

RESULTS_DIR = "results"
WRAP_WIDTH = 100


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _wrap(text, width=WRAP_WIDTH):
    """Переносит длинный текст по словам, сохраняя абзацы."""
    if not text:
        return ""
    wrapped_lines = []
    for line in str(text).split("\n"):
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.append(textwrap.fill(line, width=width))
    return "\n".join(wrapped_lines)


def save_raw_csv(results, filename="results_raw.csv"):
    """Компактный CSV: без длинного reason, только метаданные."""
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)

    # Убираем длинный reason из CSV, оставляем его только в Excel/MD
    compact = []
    for r in results:
        row = dict(r)
        row.pop("reason", None)
        row.pop("raw_answer", None)
        compact.append(row)

    if not compact:
        return

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compact[0].keys())
        writer.writeheader()
        writer.writerows(compact)
    print(f"  [save] CSV: {path}")


def save_excel_wrapped(results, filename="results_full.xlsx"):
    """Excel с переносом строк в колонках answer и reason."""
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)

    df = pd.DataFrame(results)

    # Переименуем для читаемости
    rename_map = {
        "id": "ID",
        "model": "Модель",
        "expected": "Эталон",
        "predicted": "Предсказано",
        "confidence": "Уверенность",
        "latency": "Время, с",
        "format_ok": "Формат OK",
        "reason": "Объяснение",
        "raw_answer": "Сырой ответ",
    }
    df = df.rename(columns=rename_map)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Результаты")
        worksheet = writer.sheets["Результаты"]

        # Автоподбор ширины + wrap для длинных колонок
        for col_idx, col_name in enumerate(df.columns, start=1):
            letter = worksheet.cell(row=1, column=col_idx).column_letter
            max_len = max(
                [len(str(col_name))] +
                [len(str(v)) for v in df[col_name].head(50) if v is not None]
            )
            if col_name in ("Объяснение", "Сырой ответ"):
                worksheet.column_dimensions[letter].width = 60
                for row in worksheet.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                    for cell in row:
                        cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")
            else:
                worksheet.column_dimensions[letter].width = min(max(max_len + 2, 10), 25)

        # Freeze первой строки
        worksheet.freeze_panes = "A2"

    print(f"  [save] Excel: {path}")


def save_markdown(results, filename="answers.md"):
    """Markdown-отчёт с переносами строк для чтения."""
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)

    lines = ["# Ответы моделей\n"]

    # Группируем по модели
    by_model = {}
    for r in results:
        by_model.setdefault(r.get("model", "unknown"), []).append(r)

    for model, rows in by_model.items():
        lines.append(f"\n## Модель: {model}\n")
        for r in rows:
            lines.append(f"### {r.get('id')} — эталон: `{r.get('expected')}`, предсказано: `{r.get('predicted')}`\n")
            if "confidence" in r:
                lines.append(f"- Уверенность: {r.get('confidence')}")
            if "latency" in r:
                lines.append(f"- Время: {r.get('latency'):.2f} с")
            if "format_ok" in r:
                lines.append(f"- Формат JSON OK: {r.get('format_ok')}")
            lines.append("")
            lines.append("**Вход:**")
            lines.append("```")
            lines.append(_wrap(r.get("input", "")))
            lines.append("```")
            lines.append("")
            lines.append("**Объяснение:**")
            lines.append("```")
            lines.append(_wrap(r.get("reason", "—")))
            lines.append("```")
            lines.append("")
            lines.append("**Сырой ответ:**")
            lines.append("```json")
            lines.append(_wrap(r.get("raw_answer", "—")))
            lines.append("```")
            lines.append("\n---\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [save] Markdown: {path}")


def save_all(results):
    """Сохраняет во всех трёх форматах."""
    if not results:
        print("  [save] Нет данных для сохранения")
        return
    save_raw_csv(results)
    save_excel_wrapped(results)
    save_markdown(results)