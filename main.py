import os
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
    """Извлекает JSON из ответа, даже если модель добавила текст."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class Experiment:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.system_prompt = load_prompt()
        self.tests = load_tests()
        self.models = load_models()
        self.results = []

    def ask(self, user_input):
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=TEMPERATURE,
            )
            elapsed = time.perf_counter() - started
            return response.choices[0].message.content, elapsed, None
        except Exception as e:
            elapsed = time.perf_counter() - started
            return None, elapsed, str(e)

    def run(self):
        print(SEPARATOR)
        print("ЛАБОРАТОРНАЯ РАБОТА №4: ОЦЕНКА КАЧЕСТВА LLM")
        print(SEPARATOR)
        print(f"  Моделей: {len(self.models)}")
        for m in self.models:
            print(f"    • {m}")
        print(f"  Тестов: {len(self.tests)}")
        print(f"  Температура: {TEMPERATURE}")
        print(f"  Всего запросов: {len(self.models) * len(self.tests)}")
        print(SEPARATOR)

        for model in self.models:
            self.current_model = model
            print(f"\n### МОДЕЛЬ: {model}")
            for test in self.tests:
                print(f"  [{test['id']}] ", end="", flush=True)
                raw, latency, error = self.ask(test["input"])

                parsed = extract_json(raw) if raw else None
                format_ok = parsed is not None
                predicted = parsed.get("label") if parsed else "parse_error"
                confidence = parsed.get("confidence") if parsed else None
                reason = parsed.get("reason") if parsed else None

                if error:
                    print(f"ОШИБКА ({latency:.2f} с)")
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
                    "error": error,
                    "model": model,
                })

        # Сохраняем всё
        print(f"\n{SEPARATOR}")
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print(SEPARATOR)
        save_all(self.results)

        # Метрики по каждой модели
        import pandas as pd
        df = pd.DataFrame(self.results)
        print(f"\n{SEPARATOR}")
        print("МЕТРИКИ КЛАССИФИКАЦИИ")
        print(SEPARATOR)
        for model in self.models:
            model_df = df[df["model"] == model]
            m = compute_metrics(model_df)
            print_metrics(m, model)


if __name__ == "__main__":
    exp = Experiment()
    exp.run()