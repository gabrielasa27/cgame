"""Clase de ejemplo para probar el flujo completo (panel docente + entrenamiento real).

Crea alumnos ficticios que juegan A TRAVÉS DE LA API del servidor (sus intentos pasan
por el replay y el cronómetro igual que los de un alumno de verdad) y una docente
ficticia que los evalúa. Cada alumno tiene una "habilidad" oculta que mejora con la
práctica; la docente marca "aprendió" cuando esa habilidad supera un umbral (con un 8 %
de evaluaciones que no coinciden, como pasa en un aula real).

IMPORTANTE: son datos inventados para ver cómo funciona. Todas las cuentas terminan en
@demo.cgame y se borran con `python manage.py demo --borrar`. No mezclar con datos
reales al entrenar el modelo que se usará con alumnos.
"""
from __future__ import annotations

import random
from collections import deque

from .levels import LEVELS

DEMO_DOMAIN = "@demo.cgame"
DEMO_PASSWORD = "demo1234"
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
# Orden en que los alumnos de ejemplo recorren los niveles (los más hábiles llegan más lejos)
ROUTE = ["G1-L1", "G1-L2", "G3-L1", "G2-L1", "G1-L3", "G3-L2", "G2-L2", "G3-L3", "G2-L3"]
NAMES = ["Agustina", "Benjamín", "Camila", "Dante", "Emilia", "Felipe", "Guadalupe", "Hugo",
         "Isabella", "Joaquín", "Lola", "Mateo", "Nina", "Olivia", "Pedro", "Renata", "Santino",
         "Tomás", "Uma", "Valentino", "Ximena", "Bautista", "Zoe", "Lautaro"]


def shortest_path(level) -> list[str]:
    grid = level.grid
    n = len(grid)
    cells = {(r, c): ch for r, row in enumerate(grid) for c, ch in enumerate(row)}
    start = next(p for p, ch in cells.items() if ch == "R")
    goal = next(p for p, ch in cells.items() if ch == "G")
    stars = [p for p, ch in cells.items() if ch == "*"]
    full = (1 << len(stars)) - 1
    q, par = deque([(start, 0)]), {(start, 0): None}
    while q:
        pos, m = q.popleft()
        if pos == goal and m == full:
            path, k = [], (pos, m)
            while par[k]:
                k, d = par[k]
                path.append(d)
            return path[::-1]
        for d, (dr, dc) in DIRS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or cells[nxt] == "#":
                continue
            nm = m | (1 << stars.index(nxt)) if nxt in stars else m
            if (nxt, nm) not in par:
                par[(nxt, nm)] = ((pos, m), d)
                q.append((nxt, nm))
    raise ValueError(f"nivel sin solución: {level.id}")


def play_events(level, skill: float, rng: random.Random) -> list[dict]:
    """Eventos como los que registra el juego, para un alumno con habilidad `skill` (0 a 1)."""
    t, ev = 0.0, []
    hard = 0.25 * (level.optimal_moves / 8)   # niveles largos cuestan más

    def push(kind, v=None, dt=None):
        nonlocal t
        t += dt if dt is not None else rng.uniform(1.0, 3.5) * (1.7 - skill)
        ev.append({"t": round(t, 1), "type": kind, "v": v})

    push("start", dt=0)
    fails = max(0, round((1 - skill) * 4 + hard + rng.gauss(0, 0.8)))
    hints = max(0, round((1 - skill) * 2.5 + rng.gauss(0, 0.7)))
    for i in range(hints):
        push("hint", i + 1, dt=rng.uniform(5, 15))
    if level.grid:
        path = shortest_path(level)
        for _ in range(fails):
            k = rng.randint(1, len(path) - 1)
            for d in path[:k]:
                push("add", d)
            push("run", "".join(path[:k]), dt=1)
            push("error", "incomplete", dt=0.5)
            push("clear", "".join(path[:k]))
        detours = max(0, round((1 - skill) * 3 + rng.gauss(0, 0.8)))
        for d in path:
            push("add", d)
        for _ in range(detours):  # flechas de más que luego borra (cuentan como movimientos)
            push("add", "U")
            push("remove", "U")
        push("run", "".join(path), dt=1)
    else:
        n = level.optimal_moves
        for _ in range(fails):
            order = list(range(n))
            i = rng.randrange(n - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
            for slot, c in enumerate(order):
                push("add", f"{c}@{slot + 1}")
            push("check", "".join(map(str, order)), dt=1)
            push("error", "order", dt=0.5)
            push("clear")
        for slot in range(n):
            push("add", f"{slot}@{slot + 1}")
        push("check", "".join(map(str, range(n))), dt=1)
    push("complete", dt=0.1)
    return ev


def populate(client, make_teacher, n_students: int = 24, seed: int = 2026, log=print) -> dict:
    """Usa un cliente de prueba de Flask (app.test_client()) para jugar como cada alumno.
    `make_teacher(email)` da rol docente a la cuenta de la docente de ejemplo."""
    rng = random.Random(seed)

    def account(email, name, teacher=False):
        client.post("/api/register", json={"name": name, "email": email, "password": DEMO_PASSWORD})
        if teacher:
            make_teacher(email)
        r = client.post("/api/login", json={"email": email, "password": DEMO_PASSWORD}).get_json()
        if not r or "token" not in r:
            raise RuntimeError(f"no se pudo iniciar sesión con {email}")
        return {"Authorization": "Bearer " + r["token"]}, r["user"]

    teacher_h, teacher = account("docente" + DEMO_DOMAIN, "Docente (ejemplo)", teacher=True)
    if teacher["role"] != "teacher":
        raise RuntimeError("la cuenta docente de ejemplo no tiene rol docente")
    totals = {"students": 0, "attempts": 0, "labels": 0}
    for i in range(n_students):
        name = NAMES[i % len(NAMES)] + ("" if i < len(NAMES) else f" {i // len(NAMES) + 1}")
        h, st = account(f"alumno{i + 1:02d}{DEMO_DOMAIN}", f"{name} (ejemplo)")
        base, growth = rng.uniform(0.1, 0.85), rng.uniform(0.03, 0.10)
        reach = 3 + round(base * 6)              # cuántos niveles llega a jugar
        for lid in ROUTE[:reach]:
            level = LEVELS[lid]
            skill = base - 0.06 * (level.number - 1)
            for k in range(rng.choice([1, 1, 2, 3])):
                s = min(0.98, max(0.02, skill + growth * k + rng.gauss(0, 0.07)))
                aid = client.post("/api/attempts/start", json={"level_id": lid}, headers=h).get_json()["attempt_id"]
                ev = play_events(level, s, rng)
                r = client.post("/api/attempts", headers=h, json={
                    "attempt_id": aid, "level_id": lid, "events": ev, "completed": True,
                    "time": max(1, round(ev[-1]["t"]))})
                if r.status_code != 200:
                    raise RuntimeError(f"el servidor rechazó un intento de ejemplo: {r.get_json()}")
                totals["attempts"] += 1
            # Evaluación "en el aula": depende de la habilidad oculta, no del puntaje del juego.
            cut = 0.42 + 0.06 * (level.number - 1)
            learned = s > cut
            if rng.random() < 0.08:
                learned = not learned
            client.post("/api/teacher/labels", headers=teacher_h, json={
                "student_id": st["id"], "level_id": lid, "label": int(learned),
                "note": "evaluación de ejemplo"})
            totals["labels"] += 1
        totals["students"] += 1
        log(f"  {name}: {reach} niveles")
    return totals
