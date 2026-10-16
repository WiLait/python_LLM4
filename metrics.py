from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import pandas as pd


def compute_metrics(df, positive_label="suspicious"):
    """Считает Accuracy, Precision, Recall, F1, confusion matrix."""
    y_true = df["expected"].tolist()
    y_pred = df["predicted"].tolist()

    # Убираем строки, где парсинг не удался — они считаются ошибками классификации
    valid_mask = [p in ("normal", "suspicious") for p in y_pred]
    y_true_clean = [y for y, ok in zip(y_true, valid_mask) if ok]
    y_pred_clean = [p for p, ok in zip(y_pred, valid_mask) if ok]

    if not y_true_clean:
        return {"error": "нет валидных предсказаний"}

    metrics = {
        "n_total": len(y_true),
        "n_valid": len(y_true_clean),
        "n_parse_errors": len(y_true) - len(y_true_clean),
        "accuracy": round(accuracy_score(y_true_clean, y_pred_clean), 4),
        "precision": round(precision_score(y_true_clean, y_pred_clean, pos_label=positive_label, zero_division=0), 4),
        "recall": round(recall_score(y_true_clean, y_pred_clean, pos_label=positive_label, zero_division=0), 4),
        "f1": round(f1_score(y_true_clean, y_pred_clean, pos_label=positive_label, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true_clean, y_pred_clean, labels=["normal", "suspicious"]).tolist(),
    }
    return metrics


def print_metrics(metrics, model_name):
    print(f"\n--- Метрики: {model_name} ---")
    print(f"  Всего запросов: {metrics.get('n_total')}")
    print(f"  Валидных предсказаний: {metrics.get('n_valid')}")
    print(f"  Ошибок парсинга: {metrics.get('n_parse_errors')}")
    print(f"  Accuracy:  {metrics.get('accuracy')}")
    print(f"  Precision: {metrics.get('precision')}")
    print(f"  Recall:    {metrics.get('recall')}")
    print(f"  F1:        {metrics.get('f1')}")
    cm = metrics.get("confusion_matrix")
    if cm:
        print("  Confusion matrix [[TN, FP], [FN, TP]]:")
        print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}, TP={cm[1][1]}")