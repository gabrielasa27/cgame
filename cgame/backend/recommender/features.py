"""Variables de entrada del SVM, construidas a partir de un registro de intento."""
from __future__ import annotations

FEATURES = ["accuracy", "efficiency", "speed", "errors", "hints", "level"]


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def build_features(level, accuracy: float, movements: int, time_s: float,
                   errors: int, hints_used: int) -> list[float]:
    """Todas las variables quedan aproximadamente en [0, 1]."""
    return [
        _clip(accuracy / 100.0),                                   # aciertos
        _clip(level.optimal_moves / max(movements, 1)),           # eficiencia de instrucciones
        _clip(level.par_time / max(time_s, 1)),                   # rapidez
        _clip(errors / 5.0),                                       # errores (más = peor)
        _clip(hints_used / 3.0),                                   # ayudas (más = peor)
        level.number / 3.0,                                        # dificultad del nivel
    ]


def features_from_record(levels: dict, rec: dict) -> list[float]:
    return build_features(
        levels[rec["level_id"]], rec["accuracy"], rec["movements"],
        rec["time"], rec["errors"], rec["hints_used"],
    )
