"""Carga los niveles compartidos (shared/levels.json) y calcula el óptimo de cada uno.

El cliente del juego (HTML5 / Unity) y el backend usan el mismo archivo, de modo que
`optimal_moves` y `par_time` siempre coinciden con lo que ve el estudiante.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LEVELS_PATH = Path(__file__).resolve().parents[2] / "shared" / "levels.json"


@dataclass(frozen=True)
class Level:
    id: str
    game_id: str
    number: int
    difficulty: str
    title: str
    par_time: int
    max_slots: int
    optimal_moves: int
    grid: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Game:
    id: str
    name: str
    type: str
    area: str
    description: str
    objective: str


def solve_grid(grid) -> int | None:
    """Mínimo de instrucciones (arriba/abajo/izquierda/derecha) para recoger todas
    las estrellas y terminar en la meta. Búsqueda en anchura sobre (fila, col, estrellas)."""
    n = len(grid)
    rocks, stars = set(), []
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "R":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch == "#":
                rocks.add((r, c))
            elif ch == "*":
                stars.append((r, c))
    if start is None or goal is None:
        raise ValueError("El mapa necesita una casilla R (inicio) y una G (meta)")
    full = (1 << len(stars)) - 1
    queue = deque([(start[0], start[1], 0, 0)])
    seen = {(start[0], start[1], 0)}
    while queue:
        r, c, mask, dist = queue.popleft()
        if (r, c) == goal and mask == full:
            return dist
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in rocks:
                nmask = mask
                if (nr, nc) in stars:
                    nmask |= 1 << stars.index((nr, nc))
                if (nr, nc, nmask) not in seen:
                    seen.add((nr, nc, nmask))
                    queue.append((nr, nc, nmask, dist + 1))
    return None


def load(path: Path = LEVELS_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    games = {
        g["id"]: Game(g["id"], g["name"], g["type"], g["area"], g["description"], g["objective"])
        for g in data["games"]
    }
    levels = {}
    for lv in data["levels"]:
        if "grid" in lv:
            optimal = solve_grid(lv["grid"])
            if optimal is None:
                raise ValueError(f"El nivel {lv['id']} no tiene solución")
        else:
            optimal = len(lv["steps"])
        levels[lv["id"]] = Level(
            id=lv["id"], game_id=lv["game_id"], number=lv["number"],
            difficulty=lv["difficulty"], title=lv["title"], par_time=lv["par_time"],
            max_slots=lv["max_slots"], optimal_moves=optimal,
            grid=tuple(lv.get("grid", ())),
        )
    return games, levels


GAMES, LEVELS = load()
LEVEL_ORDER = list(LEVELS.keys())  # orden pedagógico: G1-L1, G1-L2, ... G3-L3
