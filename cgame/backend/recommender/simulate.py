"""Casos simulados de desempeño.

La metodología de la investigación entrena el SVM "a partir de casos previamente
simulados". Este generador produce esos casos: cada intento parte de una habilidad
latente `s` (0 a 1); los indicadores observables (aciertos, eficiencia, rapidez, errores,
ayudas) son vistas ruidosas de esa habilidad, y la etiqueta 'aprendió' aparece cuando `s`
supera un umbral que sube con la dificultad del nivel.

IMPORTANTE: son datos artificiales. Cuando existan resultados reales de estudiantes con
una etiqueta validada por el docente, se reemplaza este módulo por esos datos.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .features import build_features
from .levels import LEVELS
from .scoring import HINT_PENALTY, PASS_SCORE, round_half_up

LABEL_FLIP = 0.02  # ruido en la etiqueta: docentes o casos límite que no encajan


def _threshold(level) -> float:
    return 0.40 + 0.08 * (level.number - 1)


def _make_attempt(level, s: float, rng: np.random.Generator) -> dict:
    acc = float(np.clip(0.30 + 0.70 * s + rng.normal(0, 0.08), 0, 1))
    eff = float(np.clip(0.25 + 0.75 * s + rng.normal(0, 0.10), 0.1, 1))
    spd = float(np.clip(0.20 + 0.80 * s + rng.normal(0, 0.12), 0.05, 1))
    errors = int(max(0, round((1 - s) * 6 + rng.normal(0, 1.0))))
    hints = int(max(0, round((1 - s) * 3 + rng.normal(0, 0.8))))
    movements = max(level.optimal_moves, round_half_up(level.optimal_moves / eff))
    time_s = max(1, round_half_up(level.par_time / spd))
    accuracy = round_half_up(acc * 100)
    efficiency = min(1.0, level.optimal_moves / movements)
    speed = min(1.0, level.par_time / time_s)
    score = max(0, min(100, round_half_up(100 * (0.5 * acc + 0.3 * efficiency + 0.2 * speed))
                       - HINT_PENALTY * hints))
    return {
        "level_id": level.id, "game_id": level.game_id, "score": score, "accuracy": accuracy,
        "time": time_s, "movements": movements, "errors": errors, "hints_used": hints,
        "completed": True, "approved": score >= PASS_SCORE,
    }


def simulate_cases(n: int, seed: int = 7):
    """Devuelve (X, y, registros): matriz de características, etiquetas y los registros crudos."""
    rng = np.random.default_rng(seed)
    levels = list(LEVELS.values())
    X, y, records = [], [], []
    for _ in range(n):
        level = levels[int(rng.integers(len(levels)))]
        s = float(rng.uniform(0, 1))
        rec = _make_attempt(level, s, rng)
        label = int(s > _threshold(level))
        if rng.random() < LABEL_FLIP:
            label = 1 - label
        rec["label"] = label
        X.append(build_features(level, rec["accuracy"], rec["movements"], rec["time"],
                                rec["errors"], rec["hints_used"]))
        y.append(label)
        records.append(rec)
    return np.array(X, dtype=float), np.array(y, dtype=int), records


def simulate_students(n_students: int = 12, attempts_each: int = 14, seed: int = 11):
    """Historiales de estudiantes que mejoran con la práctica (datos de prueba)."""
    rng = np.random.default_rng(seed)
    order = list(LEVELS.values())
    start = datetime(2026, 8, 3, 9, 0)
    rows, counter = [], 0
    for i in range(n_students):
        user_id = f"U{i + 1:03d}"
        base = float(rng.uniform(0.15, 0.6))
        growth = float(rng.uniform(0.02, 0.06))
        level_idx = 0
        attempt_no: dict[str, int] = {}
        for k in range(attempts_each):
            level = order[min(level_idx, len(order) - 1)]
            s = float(np.clip(base + growth * k + rng.normal(0, 0.06), 0.02, 0.99))
            rec = _make_attempt(level, s, rng)
            attempt_no[level.id] = attempt_no.get(level.id, 0) + 1
            counter += 1
            rec.update({
                "user_id": user_id,
                "attempt_id": f"A{counter:05d}",
                "attempt_number": attempt_no[level.id],
                "date": (start + timedelta(days=int(k // 3) + i, minutes=7 * k)).isoformat(timespec="seconds"),
            })
            rows.append(rec)
            if rec["approved"] and s > _threshold(level):
                level_idx += 1
    return rows


CSV_FIELDS = ["user_id", "game_id", "level_id", "attempt_id", "score", "accuracy", "time",
              "movements", "errors", "hints_used", "completed", "date"]


def write_sample_csv(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "completed": str(r["completed"]).lower()})
