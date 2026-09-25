"""SVM lineal: entrenamiento (scikit-learn), exportación y predicción sin dependencias."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .features import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "svm_model.json"


@dataclass
class LinearSVM:
    """Hiperplano w·z + b = 0 sobre características estandarizadas z = (x - media) / desvío.

    decision > 0  -> el estudiante 'aprendió'
    decision <= 0 -> necesita 'reforzar'
    """
    mean: list
    std: list
    w: list
    b: float
    meta: dict

    def decision(self, x) -> float:
        return sum(wi * (xi - m) / s for wi, xi, m, s in zip(self.w, x, self.mean, self.std)) + self.b

    def predict(self, x) -> int:
        return int(self.decision(x) > 0)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "LinearSVM":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["features"] != FEATURES:
            raise ValueError("El modelo guardado usa otras características; vuelve a entrenar")
        return cls(d["mean"], d["std"], d["w"], d["b"], d.get("meta", {}))

    def save(self, path: Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": FEATURES, "mean": self.mean, "std": self.std,
                   "w": self.w, "b": self.b, "meta": self.meta}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train(target: float = 0.92, max_iterations: int = 5, base_samples: int = 1500,
          seed: int = 7, verbose: bool = True):
    """Ciclo del diagrama de flujo: entrenar -> validación cruzada -> probar -> ¿92 %?

    - La validación cruzada (GridSearchCV, 5 pliegues) elige C usando SOLO datos de entrenamiento.
    - El conjunto de prueba se usa una vez por iteración solo para comprobar el objetivo.
    - Si no se cumple, no se reajusta contra la prueba: se generan más casos simulados
      (y una prueba nueva) y se repite, hasta `max_iterations`.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from .simulate import simulate_cases

    history = []
    best = None
    for it in range(1, max_iterations + 1):
        n = base_samples * it
        X, y, _ = simulate_cases(n, seed=seed + it)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
        grid = GridSearchCV(pipe, {"svc__C": [0.01, 0.1, 1, 10, 100]},
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="f1")
        grid.fit(X_tr, y_tr)
        pred = grid.predict(X_te)
        prec, rec = precision_score(y_te, pred), recall_score(y_te, pred)
        row = {"iteration": it, "samples": n, "C": grid.best_params_["svc__C"],
               "cv_f1": round(float(grid.best_score_), 4),
               "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
               "test_precision": round(float(prec), 4), "test_recall": round(float(rec), 4),
               "confusion_matrix": confusion_matrix(y_te, pred).tolist()}
        history.append(row)
        if verbose:
            print(f"[iteración {it}] n={n} C={row['C']} precisión={prec:.3f} recall={rec:.3f}")
        best = (grid, row)
        if prec >= target and rec >= target:
            break

    grid, row = best
    scaler = grid.best_estimator_.named_steps["standardscaler"]
    svc = grid.best_estimator_.named_steps["svc"]
    meta = {
        "kernel": "linear", "C": row["C"], "target": target,
        "target_met": bool(row["test_precision"] >= target and row["test_recall"] >= target),
        "test_precision": row["test_precision"], "test_recall": row["test_recall"],
        "test_accuracy": row["test_accuracy"], "samples": row["samples"],
        "iterations": len(history), "trained_on": "casos simulados (ver recommender/simulate.py)",
    }
    model = LinearSVM([float(v) for v in scaler.mean_], [float(v) for v in scaler.scale_],
                      [float(v) for v in svc.coef_[0]], float(svc.intercept_[0]), meta)
    return model, history
