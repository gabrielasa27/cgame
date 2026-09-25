"""Cálculo del resultado de un intento. Debe coincidir con `computeResult` del cliente."""
from __future__ import annotations

import math

PASS_SCORE = 60      # puntaje mínimo para aprobar un nivel
HINT_PENALTY = 5     # puntos que resta cada pista usada


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def compute_result(level, hits: int, misses: int, movements: int, time_s: float,
                   hints_used: int, completed: bool = True) -> dict:
    """hits/misses: aciertos y fallos por instrucción (o por tarjeta) durante el intento."""
    total = hits + misses
    acc_frac = hits / total if total else 0.0
    efficiency = min(1.0, level.optimal_moves / max(movements, 1))
    speed = min(1.0, level.par_time / max(time_s, 1))
    raw = 100 * (0.5 * acc_frac + 0.3 * efficiency + 0.2 * speed)
    score = max(0, min(100, round_half_up(raw) - HINT_PENALTY * hints_used))
    return {
        "accuracy": round_half_up(acc_frac * 100),
        "efficiency": efficiency,
        "speed": speed,
        "score": score,
        "approved": bool(completed and score >= PASS_SCORE),
    }
