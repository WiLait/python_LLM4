import os
import sys
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

from save import save_all
from metrics import compute_metrics, print_metrics

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
TEMPERATURE = 0.2
SEPARATOR = "=" * 70


def load_models():
    models = []
    i = 1
    while True:
        name = os.getenv(f"LLM_MODEL_{i}")
        if not name:
            break
        models.append(name)
        i += 1
    return models


def load_prompt():
    with open("prompts/prompt_v1.txt", encoding="utf-8") as f:
        return f.read().strip()


def load_tests():
    with open("tests.json", encoding="utf-8") as f:
        return json.load(f)


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


class Experiment:
    def __init__(self, model_name):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.system_prompt = load_prompt()
        self.tests = load_tests()
        self.model = model_name
        self.results = []

    def ask(self, user_input):
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - started
            usage = response.usage
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            return response.choices[0].message.content, elapsed, None, in_tok, out_tok
        except Exception as e:
            elapsed = time.perf_counter() - started
            return None, elapsed, str(e), None, None

    def run(self):
        print(SEPARATOR)
        print("ЛАБОРАТОРНАЯ РАБОТА №4: ОЦЕНКА КАЧЕСТВА LLM")
        print(SEPARATOR)
        print(f"  Модель: {self.model}")
        print(f"  Тестов: {len(self.tests)}")
        print(f"  Температура: {TEMPERATURE}")
        print(SEPARATOR)

        for test in self.tests:
            print(f"  [{test['id']}] ", end="", flush=True)
            raw, latency, error, in_tok, out_tok = self.ask(test["input"])
            parsed = extract_json(raw) if raw else None
            format_ok = parsed is not None
            predicted = parsed.get("label") if parsed else "parse_error"
            confidence = parsed.get("confidence") if parsed else None
            reason = parsed.get("reason") if parsed else None

            if error:
                print(f"ОШИБКА ({latency:.2f} с): {error[:60]}")
            else:
                mark = "✓" if predicted == test["expected"] else "✗"
                print(f"{mark} pred={predicted} ({latency:.2f} с)")

            self.results.append({
                "id": test["id"],
                "type": test["type"],
                "input": test["input"],
                "expected": test["expected"],
                "predicted": predicted,
                "confidence": confidence,
                "reason": reason,
                "latency": round(latency, 3),
                "format_ok": format_ok,
                "raw_answer": raw,
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "error": error,
                "model": self.model,
            })

        print(f"\n{SEPARATOR}")
        print("СОХРАНЕНИЕ")
        print(SEPARATOR)
        save_all(self.results)

        import pandas as pd
        df = pd.DataFrame(self.results)
        m = compute_metrics(df)
        print_metrics(m, self.model)


if __name__ == "__main__":
    models = load_models()
    if not models:
        print("Не найдено моделей в .env (LLM_MODEL_1, ...)")
        sys.exit(1)

    print("\nДоступные модели:")
    for i, m in enumerate(models, start=1):
        print(f"  [{i}] {m}")

    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        choice = input("\nВведите номер модели для теста: ").strip()

    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(models)):
            raise ValueError
    except ValueError:
        print("Неверный номер")
        sys.exit(1)

    exp = Experiment(models[idx])
    exp.run()