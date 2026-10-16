import os
import csv
import textwrap
import pandas as pd

RESULTS_DIR = "results"
WRAP_WIDTH = 90
MAX_CELL = 200  # обрезка в Excel


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _short(text, limit=MAX_CELL):
    if text is None:
        return ""
    text = str(text)
    return text[:limit] + "…" if len(text) > limit else text


def _wrap(text, width=WRAP_WIDTH):
    if not text:
        return ""
    return "\n".join(textwrap.fill(line, width=width) for line in str(text).split("\n"))


def save_raw_csv(results, filename="results_raw.csv"):
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)
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
    """Три листа: Результаты (компактно), Провалы, Метрики."""
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)

    df = pd.DataFrame(results)

    # Компактный лист — без raw_answer, с обрезкой длинных полей
    compact = df[[
        "model", "id", "type", "expected", "predicted",
        "confidence", "reason", "latency", "format_ok",
        "in_tokens", "out_tokens"
    ]].copy()
    compact["input"] = df["input"].apply(lambda t: _short(t, 120))
    compact["reason"] = compact["reason"].apply(lambda t: _short(t, 200))
    compact = compact.rename(columns={
        "model": "Модель", "id": "ID", "type": "Тип",
        "input": "Вход (сокр.)", "expected": "Эталон",
        "predicted": "Предсказано", "confidence": "Уверенность",
        "reason": "Объяснение (сокр.)", "latency": "Время, с",
        "format_ok": "Формат", "in_tokens": "Вх.ток", "out_tokens": "Вых.ток",
    })

    # Лист провалов
    failures = df[df["predicted"] != df["expected"]].copy()
    if not failures.empty:
        failures["input"] = failures["input"].apply(lambda t: _short(t, 200))
        failures["reason"] = failures["reason"].apply(lambda t: _short(t, 200))
        failures = failures[["model", "id", "type", "expected", "predicted", "reason", "input"]]
        failures = failures.rename(columns={
            "model": "Модель", "id": "ID", "type": "Тип",
            "expected": "Эталон", "predicted": "Предсказано",
            "reason": "Объяснение", "input": "Вход",
        })

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        compact.to_excel(writer, index=False, sheet_name="Результаты")
        ws = writer.sheets["Результаты"]
        widths = {"Модель": 22, "ID": 6, "Тип": 18, "Вход (сокр.)": 45,
                  "Эталон": 12, "Предсказано": 12, "Уверенность": 11,
                  "Объяснение (сокр.)": 45, "Время, с": 9,
                  "Формат": 8, "Вх.ток": 9, "Вых.ток": 10}
        for col_idx, col_name in enumerate(compact.columns, start=1):
            letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[letter].width = widths.get(col_name, 15)
        ws.freeze_panes = "A2"
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

        if not failures.empty:
            failures.to_excel(writer, index=False, sheet_name="Провалы")
            ws2 = writer.sheets["Провалы"]
            for col_idx in range(1, len(failures.columns) + 1):
                letter = ws2.cell(row=1, column=col_idx).column_letter
                ws2.column_dimensions[letter].width = 30
            ws2.freeze_panes = "A2"

    print(f"  [save] Excel: {path}")


def save_markdown(results, filename="answers.md"):
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, filename)
    lines = ["# Ответы моделей\n"]
    by_model = {}
    for r in results:
        by_model.setdefault(r.get("model", "?"), []).append(r)

    for model, rows in by_model.items():
        lines.append(f"\n## Модель: {model}\n")
        for r in rows:
            mark = "✓" if r.get("predicted") == r.get("expected") else "✗"
            lines.append(f"### {mark} {r.get('id')} — эталон: `{r.get('expected')}`, "
                         f"предсказано: `{r.get('predicted')}`\n")
            lines.append(f"- Уверенность: {r.get('confidence')}")
            lines.append(f"- Время: {r.get('latency'):.2f} с")
            lines.append(f"- Формат JSON: {r.get('format_ok')}")
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
    if not results:
        print("  [save] Нет данных")
        return
    # Имя CSV зависит от модели
    model_tag = results[0].get("model", "unknown").replace("/", "_")
    save_raw_csv(results, filename=f"results_raw_{model_tag}.csv")
    save_excel_wrapped(results, filename=f"results_full_{model_tag}.xlsx")
    save_markdown(results, filename=f"answers_{model_tag}.md")