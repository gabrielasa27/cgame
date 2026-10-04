"""Arma el conjunto de entrenamiento REAL: intentos de alumnos + etiquetas del docente.

Cada etiqueta del docente dice si un alumno domina un nivel según una evaluación hecha
fuera del juego (en el aula, en papel, por observación). Para cada etiqueta se toma el
intento completado de ese alumno en ese nivel inmediatamente ANTERIOR a la evaluación
(el que el SVM habría tenido que clasificar), y sus 6 variables pasan a ser un ejemplo
con esa etiqueta. Los intentos posteriores a la evaluación no se usan: la etiqueta no
puede "ver el futuro".

Lo usan el servidor (exportación CSV para el docente) y `recommender.train_real`.
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .features import FEATURES, features_from_record
from .levels import LEVELS

EXAMPLE_FIELDS = ["student_id", "level_id", "label", "assessed_at", "attempt_id", "attempt_date",
                  "score", "accuracy", "time", "movements", "errors", "hints_used", "verified"]


def parse_date(s) -> datetime:
    """Acepta '2026-10-03T12:00:00+00:00', '...Z' y fechas con milisegundos."""
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_examples(attempts: list[dict], labels: list[dict], *, which: str = "last",
                   verified_only: bool = True) -> tuple[list[dict], dict]:
    """Une etiquetas con intentos. which='last' (un ejemplo por etiqueta) o 'all'
    (todos los intentos del nivel anteriores a la evaluación, cada uno un ejemplo).

    Devuelve (ejemplos, resumen de lo que se descartó y por qué)."""
    by_key: dict[tuple, list[dict]] = {}
    for a in attempts:
        if not a.get("completed") or a.get("level_id") not in LEVELS:
            continue
        if verified_only and not a.get("verified"):
            continue
        by_key.setdefault((a["user_id"], a["level_id"]), []).append(a)
    for lst in by_key.values():
        lst.sort(key=lambda a: parse_date(a["date"]))

    examples, skipped = [], {"sin_intento_previo": 0, "nivel_desconocido": 0}
    for lab in labels:
        if lab["level_id"] not in LEVELS:
            skipped["nivel_desconocido"] += 1
            continue
        when = parse_date(lab["assessed_at"])
        before = [a for a in by_key.get((lab["student_id"], lab["level_id"]), [])
                  if parse_date(a["date"]) <= when]
        if not before:
            skipped["sin_intento_previo"] += 1
            continue
        for a in (before[-1:] if which == "last" else before):
            examples.append({
                "student_id": lab["student_id"], "level_id": lab["level_id"],
                "label": int(lab["label"]), "assessed_at": lab["assessed_at"],
                "attempt_id": a["attempt_id"], "attempt_date": a["date"],
                "score": a["score"], "accuracy": a["accuracy"], "time": a["time"],
                "movements": a["movements"], "errors": a["errors"],
                "hints_used": a["hints_used"], "verified": int(bool(a.get("verified"))),
            })
    return examples, skipped


def latest_labels(rows: list[dict]) -> list[dict]:
    """Cada evaluación (alumno + nivel + momento) es un ejemplo distinto y se conserva.
    Si el docente cargó dos veces la misma evaluación, vale la última que guardó."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["id"], reverse=True):
        k = (r["student_id"], r["level_id"], r["assessed_at"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[::-1]


def load_from_db(db_path: str | Path) -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(attempts)")}
        has_verified = "verified" in cols
        attempts = [dict(r) for r in conn.execute(
            "SELECT attempt_id, user_id, level_id, date, score, accuracy, time, movements, errors, "
            f"hints_used, completed{', verified' if has_verified else ', 0 AS verified'} FROM attempts")]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        labels = [dict(r) for r in conn.execute(
            "SELECT id, student_id, level_id, label, assessed_at FROM teacher_labels")] \
            if "teacher_labels" in tables else []
    finally:
        conn.close()
    return attempts, latest_labels(labels)


def write_csv(path_or_file, examples: list[dict]) -> None:
    own = isinstance(path_or_file, (str, Path))
    f = open(path_or_file, "w", newline="", encoding="utf-8") if own else path_or_file
    try:
        w = csv.DictWriter(f, fieldnames=EXAMPLE_FIELDS)
        w.writeheader()
        w.writerows(examples)
    finally:
        if own:
            f.close()


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("label", "score", "accuracy", "time", "movements", "errors", "hints_used", "verified"):
            r[k] = int(float(r[k]))
    return rows


def to_matrix(examples: list[dict]):
    """X (variables del SVM), y (etiqueta del docente), groups (alumno)."""
    import numpy as np
    X = [features_from_record(LEVELS, e) for e in examples]
    return (np.array(X, dtype=float).reshape(-1, len(FEATURES)),
            np.array([e["label"] for e in examples], dtype=int),
            np.array([e["student_id"] for e in examples]))
