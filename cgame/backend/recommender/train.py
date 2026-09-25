"""Entrena el SVM con casos simulados y exporta modelo, informe y datos de prueba.

Uso (desde la carpeta backend/):  python -m recommender.train
"""
from __future__ import annotations

import json
from pathlib import Path

from .model import MODEL_PATH, train
from .simulate import simulate_students, write_sample_csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    model, history = train()
    model.save(MODEL_PATH)
    (DATA_DIR / "training_report.json").write_text(
        json.dumps({"final": model.meta, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    write_sample_csv(DATA_DIR / "sample_attempts.csv", simulate_students())
    print("Modelo guardado en", MODEL_PATH)
    print(json.dumps(model.meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
