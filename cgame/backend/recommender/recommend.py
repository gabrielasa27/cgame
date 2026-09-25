"""Análisis del desempeño y recomendación de la siguiente actividad.

Debe coincidir con `recommend()` del cliente (client/src/index.template.html).
"""
from __future__ import annotations

from .features import features_from_record
from .levels import GAMES, LEVEL_ORDER, LEVELS
from .model import LinearSVM
from .scoring import PASS_SCORE

W_LATEST = 0.6   # peso del último intento frente al historial reciente del mismo nivel
HISTORY_N = 2    # cuántos intentos anteriores del mismo nivel se consideran

REASONS = {
    "accuracy": "Tuviste varios errores. Practica revisando cada paso.",
    "efficiency": "Usaste más pasos de los necesarios. Busca una forma más corta.",
    "speed": "Puedes practicar para resolverlo más rápido.",
    "errors": "Necesitaste varios intentos. Piensa todo el camino antes de comprobar.",
    "hints": "Necesitaste varias pistas. Intenta resolverlo con tus propias ideas.",
    "generic": "Practica un poco más para dominar este nivel.",
}


def weakest_area(f: list[float]) -> str:
    """f = [accuracy, efficiency, speed, errors, hints, level]"""
    strength = {"accuracy": f[0], "efficiency": f[1], "speed": f[2],
                "errors": 1 - f[3], "hints": 1 - f[4]}
    key = min(strength, key=strength.get)
    return key if strength[key] < 0.75 else "generic"


def _is_approved(rec: dict) -> bool:
    return bool(rec["completed"] and rec["score"] >= PASS_SCORE)


def _neighbor(level_id: str, offset: int):
    lv = LEVELS[level_id]
    target = f"{lv.game_id}-L{lv.number + offset}"
    return target if target in LEVELS else None


def recommend(model: LinearSVM, attempts: list[dict]) -> dict | None:
    """`attempts`: historial del estudiante en orden cronológico (el último es el recién terminado)."""
    done = [a for a in attempts if a["completed"]]
    if not done:
        return None
    latest = done[-1]
    f_latest = features_from_record(LEVELS, latest)
    d_last = model.decision(f_latest)
    same = [a for a in done if a["level_id"] == latest["level_id"]]
    prev = same[:-1][-HISTORY_N:]
    if prev:
        d_prev = sum(model.decision(features_from_record(LEVELS, p)) for p in prev) / len(prev)
        decision = W_LATEST * d_last + (1 - W_LATEST) * d_prev
    else:
        decision = d_last
    learned = _is_approved(latest) and decision >= 0
    level = LEVELS[latest["level_id"]]
    game = GAMES[level.game_id]

    base = {"decision": decision, "prediction": "aprendio" if learned else "refuerzo",
            "area": game.area, "game_id": game.id}
    if learned:
        nxt = _neighbor(latest["level_id"], +1)
        if nxt:
            return {**base, "level_id": nxt, "action": "advance",
                    "reason": "¡Lo dominaste! Estás listo para el siguiente reto."}
        passed = {a["level_id"] for a in done if _is_approved(a)}
        pending = next((lid for lid in LEVEL_ORDER if lid not in passed), None)
        if pending:
            g = GAMES[LEVELS[pending].game_id]
            return {**base, "level_id": pending, "action": "advance",
                    "reason": f"¡Dominaste {game.name}! Ahora prueba {g.name}."}
        best: dict[str, int] = {}
        for a in done:
            best[a["level_id"]] = max(best.get(a["level_id"], 0), a["score"])
        weakest = min(LEVEL_ORDER, key=lambda lid: best.get(lid, 0))
        return {**base, "level_id": weakest, "action": "review",
                "reason": "¡Completaste todas las misiones! Repite la que quieras mejorar."}

    # Necesita reforzar: si falló dos veces seguidas el mismo nivel, baja un escalón.
    streak = 0
    for a in reversed(same):
        if _is_approved(a) and model.decision(features_from_record(LEVELS, a)) >= 0:
            break
        streak += 1
    reason = REASONS[weakest_area(f_latest)]
    prev_level = _neighbor(latest["level_id"], -1)
    if streak >= 2 and prev_level:
        return {**base, "level_id": prev_level, "action": "step_back",
                "reason": "Vamos a practicar un nivel más fácil. " + reason}
    return {**base, "level_id": latest["level_id"], "action": "reinforce", "reason": reason}
