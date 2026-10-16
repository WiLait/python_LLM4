import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
# Модель-судья — задаётся отдельно в .env как LLM_JUDGE_MODEL
JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", os.getenv("LLM_MODEL_1"))
TEMPERATURE = 0.0
SEPARATOR = "=" * 70

RUBRIC_SYSTEM = """Ты — независимый оценщик ответов LLM.
Оцени ответ строго по данным задачи. Не учитывай красоту формулировок, если это не входит в критерии.

Критерии (шкала 0–2):
- factual_correctness: 0 — есть выдуманные факты, 1 — есть неточности, 2 — факты корректны
- completeness: 0 — ключевое пропущено, 1 — частично, 2 — всё обязательное есть
- instruction_following: 0 — формат нарушен, 1 — мелкие нарушения, 2 — полностью соблюдён
- relevance: 0 — не по теме, 1 — есть лишнее, 2 — ответ сфокусирован

Верни ТОЛЬКО JSON и кратко укажи основание каждой оценки:
{
  "factual_correctness": 0-2,
  "completeness": 0-2,
  "instruction_following": 0-2,
  "relevance": 0-2,
  "total": 0-8,
  "notes": "краткое основание"
}"""


def extract_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def build_user_prompt(test_input, expected, answer_json):
    return (
        f"### ЗАДАЧА\n"
        f"Классифицировать событие ИБ как normal или suspicious.\n\n"
        f"### ВХОД\n{test_input}\n\n"
        f"### ЭТАЛОННЫЙ ОТВЕТ\n{expected}\n\n"
        f"### ОТВЕТ МОДЕЛИ\n{answer_json}\n\n"
        f"### ТВОЯ ОЦЕНКА (только JSON):"
    )


def run_judge_on_file(csv_path):
    df = pd.read_csv(csv_path)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Берём первые 10 строк для контрольной оценки
    sample = df.head(10)
    print(SEPARATOR)
    print(f"LLM-as-a-Judge: {csv_path}")
    print(f"Судья: {JUDGE_MODEL}")
    print(f"Проверяем {len(sample)} ответов")
    print(SEPARATOR)

    rows = []
    for _, r in sample.iterrows():
        answer_json = r["raw_answer"] if "raw_answer" in r and pd.notna(r["raw_answer"]) else ""
        user_prompt = build_user_prompt(r["input"], r["expected"], answer_json)
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": RUBRIC_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            text = resp.choices[0].message.content
        except Exception as e:
            text = None
            print(f"  [{r['id']}] ошибка: {e}")
            continue
        latency = time.perf_counter() - started
        parsed = extract_json(text) or {}

        print(f"  [{r['id']}] total={parsed.get('total')} ({latency:.2f} с)")
        rows.append({
            "id": r["id"],
            "model": r["model"],
            "expected": r["expected"],
            "predicted": r["predicted"],
            "factual_correctness": parsed.get("factual_correctness"),
            "completeness": parsed.get("completeness"),
            "instruction_following": parsed.get("instruction_following"),
            "relevance": parsed.get("relevance"),
            "total": parsed.get("total"),
            "notes": parsed.get("notes"),
        })

    if rows:
        out_path = csv_path.replace(".csv", "_judge.csv").replace("results_raw_", "judge_")
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  Сохранено: {out_path}")

        avg = pd.DataFrame(rows)["total"].mean()
        print(f"  Средняя оценка судьи: {avg:.2f} / 8")


if __name__ == "__main__":
    import glob
    files = sorted(glob.glob("results/results_raw_*.csv"))
    if not files:
        print("Не найдено results_raw_*.csv. Сначала запусти main.py.")
        exit(1)
    print("Найденные файлы:")
    for f in files:
        print(f"  • {f}")
    for f in files:
        run_judge_on_file(f)