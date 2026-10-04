"""Pruebas del flujo de aprendizaje supervisado con datos reales.

Cubre: reconstrucción de intentos desde eventos (replay), puntaje recalculado en el
servidor, cronómetro del servidor, rol docente y etiquetas, migración de una base vieja,
exportación del conjunto de datos y entrenamiento separado por alumno.

Ejecutar desde backend/:  python -m unittest discover -s tests -t .   (incluye las pruebas anteriores)
"""
import json
import os
import random
import sqlite3
from contextlib import closing
import sys
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "server"))

_TMP = tempfile.mkdtemp()
os.environ["CGAME_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["CGAME_SECRET_KEY"] = "k" * 64

import app as server  # noqa: E402
from recommender.dataset import build_examples, load_from_db, read_csv, write_csv  # noqa: E402
from recommender.levels import LEVELS  # noqa: E402
from recommender.replay import ReplayError, replay  # noqa: E402
from recommender import train_real  # noqa: E402

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


# ---------- Generador de partidas realistas (eventos como los del cliente) ----------
def shortest_path(level):
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
            np_ = (pos[0] + dr, pos[1] + dc)
            if not (0 <= np_[0] < n and 0 <= np_[1] < n) or cells[np_] == "#":
                continue
            nm = m | (1 << stars.index(np_)) if np_ in stars else m
            if (np_, nm) not in par:
                par[(np_, nm)] = ((pos, m), d)
                q.append((np_, nm))


def play_events(level, skill: float, rng: random.Random) -> list:
    """Un alumno con habilidad `skill` (0-1): más fallos, pistas y tiempo cuanto menor sea."""
    t, ev = 0.0, []

    def push(kind, v=None, dt=None):
        nonlocal t
        t += dt if dt is not None else rng.uniform(1.0, 3.0) * (1.6 - skill)
        ev.append({"t": round(t, 1), "type": kind, "v": v})

    push("start", dt=0)
    fails = max(0, round((1 - skill) * 4 + rng.gauss(0, 0.7)))
    hints = max(0, round((1 - skill) * 2.5 + rng.gauss(0, 0.6)))
    for _ in range(hints):
        push("hint", 1)
    if level.grid:
        path = shortest_path(level)
        for _ in range(fails):
            k = rng.randint(1, len(path) - 1)
            for d in path[:k]:
                push("add", d)
            push("run", "".join(path[:k]), dt=1)
            push("error", "incomplete", dt=0.5)
            push("clear", "".join(path[:k]))
        extra = rng.random() < (1 - skill)
        for d in path:
            push("add", d)
        if extra:  # agrega una flecha de más y la borra (cuenta como movimiento)
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


class ApiMixin:
    @classmethod
    def setUpClass(cls):
        cls.c = server.app.test_client()

    def register(self, email, name="Alumno", role="student"):
        self.c.post("/api/register", json={"name": name, "email": email, "password": "secreto1"})
        if role != "student":
            with closing(sqlite3.connect(server.DB_PATH)) as conn, conn:
                conn.execute("UPDATE users SET role=? WHERE email=?", (role, email))
        r = self.c.post("/api/login", json={"email": email, "password": "secreto1"}).get_json()
        return {"Authorization": "Bearer " + r["token"]}, r["user"]

    def finish(self, h, level_id, events, attempt_id=None, **extra):
        body = {"level_id": level_id, "events": events, "completed": events[-1]["type"] == "complete",
                "time": max(1, round(events[-1]["t"])) if isinstance(events[-1]["t"], (int, float)) else 1,
                "attempt_id": attempt_id, **extra}
        return self.c.post("/api/attempts", json=body, headers=h)

    def start(self, h, level_id):
        return self.c.post("/api/attempts/start", json={"level_id": level_id}, headers=h).get_json()["attempt_id"]


# ---------- Replay ----------
class ReplayTest(unittest.TestCase):
    def test_perfect_grid_run(self):
        lv = LEVELS["G1-L1"]
        path = shortest_path(lv)
        ev = [{"t": 0, "type": "start"}] + [{"t": i + 1, "type": "add", "v": d} for i, d in enumerate(path)]
        ev += [{"t": 10, "type": "run", "v": "".join(path)}, {"t": 12, "type": "complete"}]
        r = replay(lv, ev)
        self.assertEqual((r.completed, r.hits, r.misses, r.movements, r.errors), (True, 5, 0, 5, 0))

    def test_bump_counts_steps_before_rock(self):
        lv = LEVELS["G1-L1"]  # R... / .#.. : derecha y luego abajo choca con la roca
        ev = [{"t": 0, "type": "start"}, {"t": 1, "type": "add", "v": "R"}, {"t": 2, "type": "add", "v": "D"},
              {"t": 3, "type": "run", "v": "RD"}]
        r = replay(lv, ev)
        self.assertEqual((r.completed, r.hits, r.misses, r.errors), (False, 1, 1, 1))

    def test_run_must_match_built_program(self):
        lv = LEVELS["G1-L1"]
        path = "".join(shortest_path(lv))
        with self.assertRaises(ReplayError):
            replay(lv, [{"t": 0, "type": "start"}, {"t": 1, "type": "add", "v": "R"}, {"t": 2, "type": "run", "v": path}])

    def test_program_cannot_exceed_slots(self):
        lv = LEVELS["G1-L1"]
        ev = [{"t": i, "type": "add", "v": "R"} for i in range(lv.max_slots + 1)]
        with self.assertRaises(ReplayError):
            replay(lv, ev)

    def test_order_level(self):
        lv = LEVELS["G3-L1"]
        ev = play_events(lv, 0.2, random.Random(1))
        r = replay(lv, ev)
        self.assertTrue(r.completed)
        self.assertEqual(r.hits + r.misses, 4 * (r.errors + 1))

    def test_events_out_of_order_rejected(self):
        lv = LEVELS["G1-L1"]
        with self.assertRaises(ReplayError):
            replay(lv, [{"t": 5, "type": "start"}, {"t": 1, "type": "add", "v": "R"}])


# ---------- Servidor ----------
class ServerTest(ApiMixin, unittest.TestCase):
    def test_score_is_recomputed_not_trusted(self):
        h, _ = self.register("a@x.com")
        lv = LEVELS["G1-L2"]
        ev = play_events(lv, 0.1, random.Random(3))
        aid = self.start(h, lv.id)
        res = self.finish(h, lv.id, ev, aid, score=100, accuracy=100, errors=0, hints_used=0).get_json()
        att = res["attempt"]
        r = replay(lv, ev)
        self.assertEqual((att["errors"], att["hints_used"], att["movements"]), (r.errors, r.hints, r.movements))
        self.assertLess(att["score"], 100)
        self.assertTrue(att["verified"])

    def test_cannot_claim_completion_without_solving(self):
        h, _ = self.register("b@x.com")
        ev = [{"t": 0, "type": "start"}, {"t": 1, "type": "add", "v": "R"}, {"t": 2, "type": "run", "v": "R"},
              {"t": 3, "type": "complete"}]
        r = self.finish(h, "G1-L1", ev, completed=True)
        self.assertEqual(r.status_code, 400)

    def test_server_clock_floors_fake_fast_time(self):
        h, _ = self.register("c@x.com")
        lv = LEVELS["G1-L1"]
        aid = self.start(h, lv.id)
        with closing(sqlite3.connect(server.DB_PATH)) as conn, conn:  # el intento empezó hace 5 minutos
            conn.execute("UPDATE attempt_sessions SET started_at=? WHERE attempt_id=?", (time.time() - 300, aid))
        ev = play_events(lv, 1.0, random.Random(4))
        att = self.finish(h, lv.id, ev, aid, time=3).get_json()["attempt"]
        self.assertGreaterEqual(att["time"], 299)

    def test_unknown_attempt_id_is_replaced_and_others_cannot_be_hijacked(self):
        ha, _ = self.register("d@x.com")
        hb, _ = self.register("e@x.com")
        aid = self.start(ha, "G1-L1")
        ev = play_events(LEVELS["G1-L1"], 0.9, random.Random(5))
        self.assertEqual(self.finish(hb, "G1-L1", ev, aid).status_code, 409)       # sesión de otro alumno
        legacy = self.finish(hb, "G1-L1", ev, "A-cliente").get_json()["attempt"]
        self.assertNotEqual(legacy["attempt_id"], "A-cliente")
        self.assertFalse(legacy["verified"])
        self.assertEqual(self.finish(ha, "G1-L1", ev, aid).status_code, 200)
        self.assertEqual(self.finish(ha, "G1-L1", ev, aid).status_code, 409)       # no se reutiliza

    def test_bad_input_is_400_not_500(self):
        h, _ = self.register("f@x.com")
        self.assertEqual(self.finish(h, "G1-L1", [{"t": "x", "type": "start"}]).status_code, 400)
        self.assertEqual(self.c.post("/api/attempts", json={"level_id": "G1-L1", "events": "x"}, headers=h).status_code, 400)
        self.assertEqual(self.c.post("/api/attempts", json={"level_id": "NOPE"}, headers=h).status_code, 400)

    def test_teacher_endpoints_and_labels(self):
        hs, st = self.register("g@x.com", "Gina")
        ht, _ = self.register("prof@x.com", "Profe", role="teacher")
        self.assertEqual(self.c.get("/api/teacher/students", headers=hs).status_code, 403)
        self.finish(hs, "G1-L1", play_events(LEVELS["G1-L1"], 0.8, random.Random(6)), self.start(hs, "G1-L1"))
        r = self.c.post("/api/teacher/labels", json={"student_id": st["id"], "level_id": "G1-L1", "label": 1},
                        headers=ht)
        self.assertEqual(r.status_code, 200)
        lab_id = r.get_json()["label"]["id"]
        future = self.c.post("/api/teacher/labels", json={"student_id": st["id"], "level_id": "G1-L1", "label": 0,
                                                          "assessed_at": "2999-01-01T00:00:00Z"}, headers=ht)
        self.assertLessEqual(future.get_json()["label"]["assessed_at"][:4], "2100")
        studs = self.c.get("/api/teacher/students", headers=ht).get_json()["students"]
        gina = next(s for s in studs if s["id"] == st["id"])
        self.assertEqual(gina["levels"]["G1-L1"]["label"]["label"], 0)   # vale la última evaluación
        csv_text = self.c.get("/api/teacher/dataset.csv", headers=ht).get_data(as_text=True)
        self.assertIn(st["id"], csv_text)
        self.assertNotIn("g@x.com", csv_text)                              # sin correos en el dataset
        self.assertEqual(self.c.delete(f"/api/teacher/labels/{lab_id}", headers=ht).status_code, 200)
        self.assertEqual(self.c.post("/api/teacher/labels", json={"student_id": st["id"], "level_id": "G1-L1",
                                                                 "label": 5}, headers=ht).status_code, 400)

    def test_me_reports_role(self):
        h, _ = self.register("prof2@x.com", role="teacher")
        self.assertEqual(self.c.get("/api/me", headers=h).get_json()["user"]["role"], "teacher")


class MigrationTest(unittest.TestCase):
    def test_old_database_is_upgraded_without_losing_rows(self):
        path = os.path.join(_TMP, "old.db")
        with closing(sqlite3.connect(path)) as conn, conn:  # esquema de la versión anterior
            conn.executescript("""
            CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL, hash TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE attempts (attempt_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, game_id TEXT NOT NULL,
                level_id TEXT NOT NULL, attempt_number INTEGER, date TEXT NOT NULL, score INTEGER, accuracy INTEGER,
                time INTEGER, movements INTEGER, errors INTEGER, hints_used INTEGER, completed INTEGER NOT NULL,
                approved INTEGER, result TEXT);
            INSERT INTO users VALUES ('U-1','Ana','ana@x.com','s','h','2026-09-01T00:00:00+00:00');
            INSERT INTO attempts VALUES ('A-1','U-1','G1','G1-L1',1,'2026-09-01T10:00:00.000Z',90,100,20,5,0,0,1,1,'Aprobado');
            """)
        server.init_db(path)
        server.init_db(path)  # idempotente
        with closing(sqlite3.connect(path)) as conn, conn:
            self.assertEqual(conn.execute("SELECT role FROM users").fetchone()[0], "student")
            self.assertEqual(conn.execute("SELECT score, verified FROM attempts").fetchone(), (90, 0))
        attempts, labels = load_from_db(path)
        self.assertEqual((len(attempts), labels), (1, []))


# ---------- Entrenamiento con datos reales ----------
class TrainRealTest(ApiMixin, unittest.TestCase):
    """Una 'clase' simulada juega a través de la API real; el docente etiqueta según una
    habilidad que el modelo no ve. El script debe aprender algo y no mezclar alumnos."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.db = os.path.join(_TMP, "class.db")
        server.DB_PATH = cls.db
        server.init_db(cls.db)
        rng = random.Random(42)
        cls.ht, _ = ApiMixin.register(cls, "docente@x.com", "Docente", role="teacher")
        for i in range(24):
            h, st = ApiMixin.register(cls, f"al{i}@x.com", f"Alumno {i}")
            base = rng.uniform(0.05, 0.95)
            for lid in ("G1-L1", "G1-L2", "G2-L1", "G3-L2"):
                skill = min(1, max(0, base + rng.gauss(0, 0.08)))
                aid = ApiMixin.start(cls, h, lid)
                ApiMixin.finish(cls, h, lid, play_events(LEVELS[lid], skill, rng), aid)
                cut = 0.45 + 0.05 * LEVELS[lid].number
                label = int(skill > cut) if rng.random() > 0.05 else int(skill <= cut)
                cls.c.post("/api/teacher/labels", json={"student_id": st["id"], "level_id": lid, "label": label},
                           headers=cls.ht)
        # una evaluación ANTERIOR a cualquier intento: no debe usarse
        cls.c.post("/api/teacher/labels", json={"student_id": st["id"], "level_id": "G3-L3", "label": 1},
                   headers=cls.ht)

    @classmethod
    def tearDownClass(cls):
        server.DB_PATH = os.environ["CGAME_DB_PATH"]

    def test_examples_join_and_skip(self):
        attempts, labels = load_from_db(self.db)
        ex, skipped = build_examples(attempts, labels)
        self.assertEqual(len(ex), 24 * 4)
        self.assertEqual(skipped["sin_intento_previo"], 1)

    def test_training_separates_students_and_reports(self):
        attempts, labels = load_from_db(self.db)
        ex, _ = build_examples(attempts, labels)
        model, report = train_real.run(ex, min_students=8)
        self.assertEqual(report["train_students"] + report["test_students"], 24)
        self.assertGreater(report["test"]["nuevo_svm"]["balanced_accuracy"], 0.6)
        self.assertIn("datos reales", model.meta["trained_on"])
        self.assertEqual(len(model.w), 6)

    def test_csv_round_trip(self):
        attempts, labels = load_from_db(self.db)
        ex, _ = build_examples(attempts, labels)
        p = os.path.join(_TMP, "ds.csv")
        write_csv(p, ex)
        self.assertEqual(read_csv(p), [{**e, **{k: int(e[k]) for k in ("label", "score", "accuracy", "time",
                         "movements", "errors", "hints_used", "verified")}} for e in ex])

    def test_not_enough_data_is_reported(self):
        attempts, labels = load_from_db(self.db)
        ex, _ = build_examples(attempts, labels)
        few = [e for e in ex if e["student_id"] in {e["student_id"] for e in ex[:8]}]
        with self.assertRaises(train_real.NotEnoughData):
            train_real.run(few, min_students=8)

    def test_cli_does_not_install_unless_asked(self):
        before = (BACKEND / "data" / "svm_model.json").read_text(encoding="utf-8")
        code = train_real.main(["--db", self.db, "--out-dir", _TMP])
        self.assertEqual(code, 0)
        self.assertEqual((BACKEND / "data" / "svm_model.json").read_text(encoding="utf-8"), before)
        self.assertTrue(Path(_TMP, "svm_model_real.json").exists())


if __name__ == "__main__":
    unittest.main()
