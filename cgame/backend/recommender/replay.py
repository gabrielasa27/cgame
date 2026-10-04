"""Reconstrucción de un intento a partir de sus eventos crudos (lado servidor).

El cliente ya registra cada acción del estudiante en `events` (agregar/quitar una
instrucción, ejecutar, comprobar, pedir pista...). En vez de confiar en el puntaje que
calcula el navegador, el servidor vuelve a "jugar" esos eventos sobre el mismo mapa
(`shared/levels.json`) y obtiene aciertos, fallos, movimientos, errores y pistas.

Así un intento solo cuenta como completado si el programa que el estudiante armó
realmente lleva a Cody a la meta (o si el orden de tarjetas es realmente el correcto).
Debe reproducir exactamente la lógica de `executeProgram` y `checkOrder` del cliente.
"""
from __future__ import annotations

from dataclasses import dataclass

from .levels import Level

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MAX_EVENTS = 5000          # un intento normal tiene decenas de eventos
KNOWN = {"start", "add", "remove", "clear", "run", "check", "error", "hint",
         "idle_nudge", "complete", "abandon"}


class ReplayError(ValueError):
    """Los eventos no corresponden a un intento posible en este nivel."""


@dataclass
class Replay:
    hits: int = 0
    misses: int = 0
    movements: int = 0
    errors: int = 0
    hints: int = 0
    completed: bool = False
    last_t: float = 0.0     # segundo (según el reloj del cliente) del último evento


def parse_grid(rows):
    rocks, stars, start, goal = set(), [], None, None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "R":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch == "#":
                rocks.add((r, c))
            elif ch == "*":
                stars.append((r, c))
    return len(rows), start, goal, rocks, stars


def run_program(level: Level, program: str) -> tuple[int, bool]:
    """Ejecuta el programa. Devuelve (pasos válidos antes de chocar, ¿llegó a la meta con todas las estrellas?)."""
    n, pos, goal, rocks, stars = parse_grid(level.grid)
    got = set()
    ok_steps = 0
    for d in program:
        nr, nc = pos[0] + DIRS[d][0], pos[1] + DIRS[d][1]
        if not (0 <= nr < n and 0 <= nc < n) or (nr, nc) in rocks:
            return ok_steps, False
        pos = (nr, nc)
        if pos in stars:
            got.add(pos)
        ok_steps += 1
    return ok_steps, pos == goal and len(got) == len(stars)


def _time(ev) -> float:
    try:
        t = float(ev.get("t", 0))
    except (TypeError, ValueError):
        raise ReplayError("tiempo de evento inválido")
    if t < 0 or t != t:
        raise ReplayError("tiempo de evento inválido")
    return t


def replay(level: Level, events: list) -> Replay:
    if not isinstance(events, list) or not events:
        raise ReplayError("faltan los eventos del intento")
    if len(events) > MAX_EVENTS:
        raise ReplayError("demasiados eventos")
    out = Replay()
    is_grid = bool(level.grid)
    n_cards = level.optimal_moves  # en "Ordena los Pasos" el óptimo es la cantidad de tarjetas
    program: list[str] = []
    slots: list[int | None] = [None] * n_cards
    prev_t = 0.0

    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") not in KNOWN:
            raise ReplayError("evento desconocido")
        if out.completed and ev["type"] not in ("complete", "idle_nudge"):
            raise ReplayError("eventos después de terminar el nivel")
        t = _time(ev)
        if t + 0.05 < prev_t:
            raise ReplayError("los eventos no están en orden")
        prev_t = max(prev_t, t)
        kind, v = ev["type"], ev.get("v")

        if kind == "add":
            out.movements += 1
            if is_grid:
                if v not in DIRS:
                    raise ReplayError("instrucción inválida")
                if len(program) >= level.max_slots:
                    raise ReplayError("programa más largo de lo permitido")
                program.append(v)
            else:
                card, slot = _card(v, n_cards)
                if slot != _first_free(slots) or card in slots:
                    raise ReplayError("tarjeta colocada en un lugar imposible")
                slots[slot] = card
        elif kind == "remove":
            if is_grid:
                if not program or program[-1] != v:
                    raise ReplayError("se quitó una instrucción que no estaba")
                program.pop()
            else:
                card, slot = _card(v, n_cards)
                if slots[slot] != card:
                    raise ReplayError("se quitó una tarjeta que no estaba")
                slots[slot] = None
        elif kind == "clear":
            program, slots = [], [None] * n_cards
        elif kind == "run":
            if not is_grid or v != "".join(program) or not program:
                raise ReplayError("el programa ejecutado no coincide con el armado")
            ok_steps, success = run_program(level, program)
            out.hits += ok_steps
            if success:
                out.completed = True
            else:
                out.misses += 1
                out.errors += 1
        elif kind == "check":
            if is_grid or None in slots or v != "".join(str(c) for c in slots):
                raise ReplayError("el orden comprobado no coincide con el armado")
            ok = sum(1 for i, c in enumerate(slots) if c == i)
            out.hits += ok
            out.misses += n_cards - ok
            if ok == n_cards:
                out.completed = True
            else:
                out.errors += 1
        elif kind == "hint":
            out.hints += 1
        # start, error, idle_nudge, complete, abandon: solo marcas de tiempo
    out.last_t = prev_t
    return out


def _card(v, n: int) -> tuple[int, int]:
    try:
        card, slot = str(v).split("@")
        card, slot = int(card), int(slot) - 1
    except (ValueError, AttributeError):
        raise ReplayError("tarjeta inválida")
    if not (0 <= card < n and 0 <= slot < n):
        raise ReplayError("tarjeta inválida")
    return card, slot


def _first_free(slots) -> int:
    return slots.index(None) if None in slots else -1
