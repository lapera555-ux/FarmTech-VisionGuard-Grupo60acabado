from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


DETECTOR_COLUMNS = [
    "modelo", "tipo", "precision", "recall", "f1", "map50",
    "ap50_person", "ap50_car", "mean_inference_ms", "training_seconds",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_comparison(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = []

    files = [
        ("YOLOv5 custom - 30 épocas", "detecção", "yolo_custom_30_metrics.json"),
        ("YOLOv5 custom - 60 épocas", "detecção", "yolo_custom_60_metrics.json"),
        ("YOLOv5 padrão COCO", "detecção", "yolo_standard_metrics.json"),
        ("CNN do zero", "classificação", "cnn_metrics.json"),
    ]

    for name, kind, filename in files:
        path = results_dir / filename
        if not path.exists():
            continue
        data = load_json(path)
        data["modelo"] = name
        data["tipo"] = kind
        rows.append(data)

    if not rows:
        return pd.DataFrame()
    preferred = ["modelo", "tipo", "accuracy", "precision", "recall", "f1", "map50",
                 "ap50_person", "ap50_car", "mean_inference_ms", "training_seconds"]
    df = pd.DataFrame(rows)
    cols = [c for c in preferred if c in df.columns]
    return df[cols]


def save_comparison(results_dir: str | Path) -> pd.DataFrame:
    df = build_comparison(results_dir)
    if not df.empty:
        df.to_csv(Path(results_dir) / "metrics_summary.csv", index=False)
    return df
