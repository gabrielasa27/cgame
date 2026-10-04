"""Entrena el SVM con datos REALES: intentos guardados en cgame.db + etiquetas del docente.

Uso (desde la carpeta backend/):
    python -m recommender.train_real --db server/cgame.db            # solo evalúa e informa
    python -m recommender.train_real --db server/cgame.db --install  # además lo instala en el juego
    python -m recommender.train_real --csv cgame_dataset.csv          # desde el CSV del panel docente

Cómo evita inflar los resultados:
  * La separación entrenamiento/prueba es POR ALUMNO (StratifiedGroupKFold): ningún alumno
    tiene intentos en los dos lados. Si no, el modelo "reconocería" a un alumno que ya vio
    y la precisión saldría más alta de lo que sería con alumnos nuevos.
  * El hiperparámetro C (y el peso de clases) se elige con validación cruzada también
    agrupada por alumno, usando SOLO los alumnos de entrenamiento.
  * Los alumnos de prueba se usan una sola vez, al final, para medir.
  * Se compara contra lo que el juego hace hoy (SVM entrenado con casos simulados + regla
    de puntaje ≥ 60) con los mismos alumnos de prueba. Con --install solo se reemplaza el
    modelo si el nuevo es mejor (o si se fuerza con --force).
  * Por defecto solo se usan intentos verificados (puntaje recalculado y cronometrado por
    el servidor) y, por cada evaluación del docente, el último intento ANTERIOR a ella.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .dataset import build_examples, load_from_db, read_csv, to_matrix
from .model import MODEL_PATH, LinearSVM
from .scoring import PASS_SCORE

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "server" / "cgame.db"
GRID = {"svc__C": [0.01, 0.1, 1, 10, 100], "svc__class_weight": [None, "balanced"]}


class NotEnoughData(Exception):
    pass


def metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        # filas = etiqueta del docente (0 no aprendió, 1 aprendió); columnas = predicción
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def check_enough(y, groups, min_students: int, min_per_class: int) -> None:
    n_students = len(set(groups))
    problems = []
    if n_students < min_students:
        problems.append(f"hay {n_students} alumnos con evaluaciones; se necesitan al menos {min_students}")
    for cls, name in ((1, "aprendió"), (0, "no aprendió")):
        n = int((y == cls).sum())
        if n < min_per_class:
            problems.append(f"hay {n} ejemplos '{name}'; se necesitan al menos {min_per_class}")
    if problems:
        raise NotEnoughData("; ".join(problems))


def split_by_student(y, groups, test_size: float, seed: int):
    """Separa alumnos (no intentos) en entrenamiento y prueba, intentando que ambos lados
    tengan las dos clases."""
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, round(1 / test_size))
    for s in range(seed, seed + 50):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        tr, te = next(cv.split(np.zeros(len(y)), y, groups))
        if len(set(y[tr])) == 2 and len(set(y[te])) == 2:
            return tr, te
    raise NotEnoughData("no se pudo separar alumnos de modo que entrenamiento y prueba tengan "
                        "ejemplos de las dos clases; hacen falta más alumnos evaluados")


def fit_grid(X, y, groups, seed: int):
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    n_splits = min(5, len(set(groups)))
    pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    grid = GridSearchCV(pipe, GRID, scoring="balanced_accuracy",
                        cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    grid.fit(X, y, groups=groups)
    return grid


def to_linear_svm(grid, meta: dict) -> LinearSVM:
    scaler = grid.best_estimator_.named_steps["standardscaler"]
    svc = grid.best_estimator_.named_steps["svc"]
    return LinearSVM([float(v) for v in scaler.mean_], [float(v) for v in scaler.scale_],
                     [float(v) for v in svc.coef_[0]], float(svc.intercept_[0]), meta)


def run(examples: list[dict], *, test_size: float = 0.25, seed: int = 7, min_students: int = 8,
        min_per_class: int = 5, current: LinearSVM | None = None) -> tuple[LinearSVM, dict]:
    X, y, groups = to_matrix(examples)
    check_enough(y, groups, min_students, min_per_class)
    tr, te = split_by_student(y, groups, test_size, seed)
    assert not set(groups[tr]) & set(groups[te]), "un alumno quedó en entrenamiento y prueba"

    grid = fit_grid(X[tr], y[tr], groups[tr], seed)
    pred_new = grid.predict(X[te])

    current = current or LinearSVM.load(MODEL_PATH)
    scores = np.array([e["score"] for e in examples])
    d_cur = np.array([current.decision(x) for x in X[te]])
    baselines = {
        "regla_puntaje_60": metrics(y[te], (scores[te] >= PASS_SCORE).astype(int)),
        "svm_actual": metrics(y[te], (d_cur >= 0).astype(int)),
        # lo que hoy decide el juego para 'aprendió': aprobó Y el SVM actual da >= 0
        "juego_actual": metrics(y[te], ((scores[te] >= PASS_SCORE) & (d_cur >= 0)).astype(int)),
    }
    new = metrics(y[te], pred_new)

    # Modelo final: mismos hiperparámetros elegidos por CV, entrenado con TODOS los alumnos.
    final_grid = fit_grid(X, y, groups, seed)
    meta = {
        "kernel": "linear", "C": final_grid.best_params_["svc__C"],
        "class_weight": final_grid.best_params_["svc__class_weight"],
        "trained_on": f"datos reales: {len(set(groups))} alumnos, {len(y)} evaluaciones del docente",
        "samples": int(len(y)), "students": int(len(set(groups))),
        "test_students": int(len(set(groups[te]))),
        "test_precision": new["precision"], "test_recall": new["recall"], "test_accuracy": new["accuracy"],
        "test_balanced_accuracy": new["balanced_accuracy"],
        "cv_balanced_accuracy": round(float(final_grid.best_score_), 4),
        "split": "por alumno (StratifiedGroupKFold)",
    }
    report = {
        "examples": int(len(y)), "students": int(len(set(groups))),
        "positives": int(y.sum()), "negatives": int((1 - y).sum()),
        "train_students": int(len(set(groups[tr]))), "test_students": int(len(set(groups[te]))),
        "test_examples": int(len(te)),
        "best_params_train": {k.replace("svc__", ""): v for k, v in grid.best_params_.items()},
        "cv_balanced_accuracy_train": round(float(grid.best_score_), 4),
        "test": {"nuevo_svm": new, **baselines},
        "weights": dict(zip(["accuracy", "efficiency", "speed", "errors", "hints", "level"],
                            [round(float(w), 4) for w in final_grid.best_estimator_.named_steps["svc"].coef_[0]])),
    }
    return to_linear_svm(final_grid, meta), report


def print_report(report: dict) -> None:
    print(f"\nEjemplos: {report['examples']} de {report['students']} alumnos "
          f"(aprendió {report['positives']} / no aprendió {report['negatives']})")
    print(f"Entrenamiento: {report['train_students']} alumnos · Prueba: {report['test_students']} alumnos "
          f"({report['test_examples']} evaluaciones) — ningún alumno está en los dos grupos")
    print(f"Hiperparámetros elegidos por CV agrupada: {report['best_params_train']}\n")
    names = {"nuevo_svm": "SVM con datos reales", "juego_actual": "Juego actual (SVM simulado + puntaje≥60)",
             "svm_actual": "SVM simulado solo", "regla_puntaje_60": "Solo regla puntaje ≥ 60"}
    print(f"{'Con alumnos de prueba':42} {'exact.':>7} {'bal.':>7} {'prec.':>7} {'recall':>7} {'F1':>7}")
    for k, m in report["test"].items():
        print(f"{names[k]:42} {m['accuracy']:7.3f} {m['balanced_accuracy']:7.3f} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f}")
    if report["test_students"] < 5:
        print("\nAVISO: hay muy pocos alumnos de prueba; estas cifras pueden cambiar mucho con más datos.")
    print("\nPesos del modelo final (positivo = empuja hacia 'aprendió'):")
    for k, w in report["weights"].items():
        print(f"  {k:10} {w:+.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--db", type=Path, default=None, help=f"ruta a cgame.db (por defecto {DEFAULT_DB})")
    src.add_argument("--csv", type=Path, help="CSV descargado del panel docente")
    ap.add_argument("--which", choices=["last", "all"], default="last",
                    help="por evaluación: solo el último intento previo (por defecto) o todos los previos")
    ap.add_argument("--include-unverified", action="store_true",
                    help="incluir intentos sin cronómetro del servidor (por ejemplo, anteriores a esta versión)")
    ap.add_argument("--test-size", type=float, default=0.25, help="fracción de ALUMNOS para la prueba")
    ap.add_argument("--min-students", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--install", action="store_true",
                    help="reemplazar data/svm_model.json si el nuevo modelo es mejor que el actual")
    ap.add_argument("--force", action="store_true", help="con --install, instalar aunque no sea mejor")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR, help="dónde guardar el modelo y el informe")
    args = ap.parse_args(argv)

    if args.csv:
        examples = read_csv(args.csv)
        if not args.include_unverified:
            examples = [e for e in examples if e["verified"]]
    else:
        attempts, labels = load_from_db(args.db or DEFAULT_DB)
        examples, skipped = build_examples(attempts, labels, which=args.which,
                                           verified_only=not args.include_unverified)
        if skipped["sin_intento_previo"]:
            print(f"Se ignoraron {skipped['sin_intento_previo']} evaluaciones sin un intento "
                  f"{'' if args.include_unverified else 'verificado '}previo en ese nivel.")
    try:
        model, report = run(examples, test_size=args.test_size, seed=args.seed, min_students=args.min_students)
    except NotEnoughData as e:
        print(f"Todavía no hay datos suficientes para entrenar con datos reales: {e}.")
        print("Sigue juntando intentos y evaluaciones (python server/manage.py stats para ver el avance).")
        return 2

    print_report(report)
    out = args.out_dir / "svm_model_real.json"
    model.save(out)
    (args.out_dir / "real_training_report.json").write_text(
        json.dumps({"final": model.meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModelo guardado en {out} (informe: {out.parent / 'real_training_report.json'})")

    if args.install:
        new_b, cur_b = report["test"]["nuevo_svm"]["balanced_accuracy"], report["test"]["juego_actual"]["balanced_accuracy"]
        if new_b <= cur_b and not args.force:
            print(f"\nNO se instaló: con los alumnos de prueba el nuevo modelo ({new_b:.3f}) no supera "
                  f"al actual ({cur_b:.3f}) en exactitud balanceada. Usa --force si igual lo quieres.")
            return 0
        backup = DATA_DIR / "svm_model.simulado.json"
        if not backup.exists():
            shutil.copy(MODEL_PATH, backup)
        model.save(MODEL_PATH)
        print(f"\nInstalado en {MODEL_PATH} (el modelo simulado quedó en {backup.name}).")
        print("Siguiente paso: recargar la app web del servidor (en PythonAnywhere: pestaña Web → Reload).")
        print("Opcional: python tools/build_client.py para que el modo sin servidor use el mismo modelo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
