"""Servidor real de CGAME (API que espera client/src/index.template.html vía ?api=).

Envuelve exactamente el mismo código de backend/recommender/ que ya se probó y
entrenó (compute_result, recommend, LinearSVM) — no reimplementa esa lógica.

Ejecutar en desarrollo:
    cd backend/server
    pip install -r requirements.txt
    python app.py                      # sirve en http://127.0.0.1:5057

Variables de entorno:
    CGAME_SECRET_KEY   clave para firmar los tokens (genera una propia en producción)
    CGAME_DB_PATH      ruta del archivo SQLite (por defecto: server/cgame.db)
    CGAME_CORS_ORIGIN  origen permitido para CORS (por defecto: *)
    PORT               puerto (por defecto: 5057)

Datos confiables para el aprendizaje supervisado:
  * El puntaje NO se toma del navegador. El cliente envía los eventos crudos del intento
    (cada instrucción, cada ejecución, cada pista) y el servidor los vuelve a jugar sobre
    el mapa real (recommender/replay.py) para obtener aciertos, fallos, movimientos,
    errores y pistas; luego calcula el puntaje con recommender/scoring.py.
  * El servidor cronometra el intento: el cliente pide POST /api/attempts/start al
    empezar, y al terminar el tiempo no puede ser menor al que midió el servidor.
    Los intentos con sesión del servidor quedan marcados `verified = 1`.
  * Los docentes (rol 'teacher', ver manage.py) cargan etiquetas "aprendió / no
    aprendió" por alumno y nivel; recommender/train_real.py entrena con ellas.
Límite honesto: un estudiante con conocimientos técnicos podría programar un bot que
juegue por él; eso no se puede impedir del lado del servidor, pero ya no puede inventar
un puntaje ni un intento imposible.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import jwt
from flask import Flask, Response, g, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/  -> import recommender
from recommender.dataset import build_examples, latest_labels, parse_date, write_csv  # noqa: E402
from recommender.levels import LEVEL_ORDER, LEVELS  # noqa: E402
from recommender.model import LinearSVM  # noqa: E402
from recommender.recommend import recommend  # noqa: E402
from recommender.replay import ReplayError, replay  # noqa: E402
from recommender.scoring import compute_result  # noqa: E402

SECRET_KEY = os.environ.get("CGAME_SECRET_KEY") or secrets.token_hex(32)
DB_PATH = os.environ.get("CGAME_DB_PATH", str(Path(__file__).resolve().parent / "cgame.db"))
CORS_ORIGIN = os.environ.get("CGAME_CORS_ORIGIN", "*")
TOKEN_DAYS = 30
SESSION_MAX_AGE = 4 * 3600   # un intento abierto más de 4 h ya no se considera cronometrado
TIME_SLACK = 10              # segundos de tolerancia entre el reloj del cliente y el del servidor

if not os.environ.get("CGAME_SECRET_KEY"):
    print("[cgame] AVISO: CGAME_SECRET_KEY no está definida; se generó una clave temporal "
          "que cambiará cada reinicio (los tokens dejarán de servir). Defínela en producción.",
          file=sys.stderr)

app = Flask(__name__)
MODEL = LinearSVM.load()  # mismos coeficientes que ve el juego (svm_model.json)


# ---------- Base de datos ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=15)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def _add_column(conn, table: str, column: str, ddl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db(path: str = DB_PATH):
    """Crea las tablas y migra una base existente sin perder datos (solo agrega columnas)."""
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL, hash TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
            game_id TEXT NOT NULL, level_id TEXT NOT NULL, attempt_number INTEGER,
            date TEXT NOT NULL, score INTEGER, accuracy INTEGER, time INTEGER,
            movements INTEGER, errors INTEGER, hints_used INTEGER,
            completed INTEGER NOT NULL, approved INTEGER, result TEXT
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(id),
            attempt_id TEXT, level_id TEXT, action TEXT, prediction TEXT, area TEXT,
            game_id TEXT, decision REAL, reason TEXT, date TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempt_sessions (
            attempt_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
            level_id TEXT NOT NULL, started_at REAL NOT NULL, finished INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS teacher_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL REFERENCES users(id),
            level_id TEXT NOT NULL,
            label INTEGER NOT NULL CHECK (label IN (0, 1)),   -- 1 = aprendió, 0 = no aprendió
            note TEXT,
            teacher_id TEXT NOT NULL REFERENCES users(id),
            assessed_at TEXT NOT NULL,                         -- cuándo se evaluó (fuera del juego)
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, date);
        CREATE INDEX IF NOT EXISTS idx_labels_student ON teacher_labels(student_id, level_id);
        """)
        _add_column(conn, "users", "role", "TEXT NOT NULL DEFAULT 'student'")
        _add_column(conn, "attempts", "hits", "INTEGER")
        _add_column(conn, "attempts", "misses", "INTEGER")
        _add_column(conn, "attempts", "events", "TEXT")          # eventos crudos (JSON) para investigación
        _add_column(conn, "attempts", "verified", "INTEGER NOT NULL DEFAULT 0")
        _add_column(conn, "attempts", "client_time", "INTEGER")
        _add_column(conn, "attempts", "server_time", "INTEGER")
        conn.commit()


# ---------- Utilidades ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def make_token(user_id: str) -> str:
    payload = {"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + TOKEN_DAYS * 86400}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def error(code: str, status: int, detail: str | None = None):
    body = {"error": code}
    if detail:
        body["detail"] = detail
    return jsonify(body), status


def require_auth(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return error("unauthorized", 401)
        try:
            payload = jwt.decode(auth[7:], SECRET_KEY, algorithms=["HS256"])
        except jwt.PyJWTError:
            return error("unauthorized", 401)
        row = db().execute("SELECT * FROM users WHERE id=?", (payload["sub"],)).fetchone()
        if not row:
            return error("unauthorized", 401)
        g.user = row
        return fn(*a, **kw)
    return wrapper


def require_teacher(fn):
    @require_auth
    @wraps(fn)
    def wrapper(*a, **kw):
        if g.user["role"] != "teacher":
            return error("forbidden", 403)
        return fn(*a, **kw)
    return wrapper


def public_user(row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"],
            "created_at": row["created_at"], "role": row["role"]}


PUBLIC_ATTEMPT = ["attempt_id", "user_id", "game_id", "level_id", "attempt_number", "date", "score",
                  "accuracy", "time", "movements", "errors", "hints_used", "completed", "approved",
                  "result", "verified"]


def row_to_attempt(row) -> dict:
    d = {k: row[k] for k in PUBLIC_ATTEMPT}
    for k in ("completed", "approved", "verified"):
        d[k] = bool(d[k])
    return d


def user_attempts(user_id: str) -> list[dict]:
    rows = db().execute("SELECT * FROM attempts WHERE user_id=? ORDER BY date ASC", (user_id,)).fetchall()
    return [row_to_attempt(r) for r in rows]


def as_int(value, lo: int = 0, hi: int = 10**6) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


# ---------- CORS ----------
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Vary"] = "Origin"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------- Rutas públicas ----------
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL.meta, "levels": len(LEVELS)})


@app.get("/api/model")
def model_coefficients():
    """El cliente usa estos coeficientes para mostrar 'cómo decidió el SVM' con el modelo
    que realmente corre en el servidor (por ejemplo, uno reentrenado con datos reales)."""
    return jsonify({"features": ["accuracy", "efficiency", "speed", "errors", "hints", "level"],
                    "mean": MODEL.mean, "std": MODEL.std, "w": MODEL.w, "b": MODEL.b, "meta": MODEL.meta})


@app.post("/api/register")
def register():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()[:80]
    email = str(body.get("email") or "").strip().lower()[:200]
    password = str(body.get("password") or "")
    if not name or "@" not in email or len(password) < 6:
        return error("invalid_input", 400)
    if db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        return error("email_taken", 409)
    salt = secrets.token_hex(16)
    user_id = "U-" + secrets.token_hex(8)
    created = now_iso()
    db().execute("INSERT INTO users (id,name,email,salt,hash,created_at,role) VALUES (?,?,?,?,?,?,'student')",
                 (user_id, name, email, salt, hash_password(password, salt), created))
    db().commit()
    return jsonify({"user": {"id": user_id, "name": name, "email": email, "created_at": created, "role": "student"}})


@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    email, password = str(body.get("email") or "").strip().lower(), str(body.get("password") or "")
    urow = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not urow or not secrets.compare_digest(hash_password(password, urow["salt"]), urow["hash"]):
        return error("bad_credentials", 401)
    return jsonify({"user": public_user(urow), "token": make_token(urow["id"])})


# ---------- Alumno ----------
@app.get("/api/me")
@require_auth
def me():
    return jsonify({"user": public_user(g.user)})


@app.get("/api/attempts")
@require_auth
def list_attempts():
    return jsonify({"attempts": user_attempts(g.user["id"])})


@app.post("/api/attempts/start")
@require_auth
def start_attempt():
    """Abre un intento cronometrado por el servidor. Devuelve el attempt_id a usar al terminar."""
    body = request.get_json(silent=True) or {}
    level_id = body.get("level_id")
    if level_id not in LEVELS:
        return error("invalid_level", 400)
    attempt_id = "A-" + secrets.token_hex(8)
    db().execute("INSERT INTO attempt_sessions (attempt_id,user_id,level_id,started_at) VALUES (?,?,?,?)",
                 (attempt_id, g.user["id"], level_id, time.time()))
    db().commit()
    return jsonify({"attempt_id": attempt_id})


@app.post("/api/attempts")
@require_auth
def save_attempt():
    body = request.get_json(silent=True) or {}
    level_id = body.get("level_id")
    if level_id not in LEVELS:
        return error("invalid_level", 400)
    level = LEVELS[level_id]

    # 1) Reconstruir el intento desde los eventos crudos (no se confía en el puntaje del cliente).
    try:
        rp = replay(level, body.get("events"))
    except ReplayError as e:
        return error("invalid_events", 400, str(e))
    wants_complete = bool(body.get("completed"))
    if wants_complete and not rp.completed:
        return error("invalid_events", 400, "el intento no llega a resolver el nivel")
    completed = rp.completed

    # 2) Tiempo: el del cliente, pero nunca menor al que midió el servidor.
    client_time = as_int(body.get("time"), 0, 24 * 3600)
    if client_time is None:
        client_time = max(1, round(rp.last_t))
    server_time, verified, date = None, False, None
    session = None
    sid = body.get("attempt_id")
    if isinstance(sid, str):
        session = db().execute("SELECT * FROM attempt_sessions WHERE attempt_id=?", (sid,)).fetchone()
    if session is not None:
        if session["user_id"] != g.user["id"] or session["level_id"] != level_id or session["finished"]:
            return error("invalid_attempt", 409)
        elapsed = time.time() - session["started_at"]
        if elapsed <= SESSION_MAX_AGE:
            server_time, verified = round(elapsed), True
        date = datetime.fromtimestamp(session["started_at"], timezone.utc).isoformat(timespec="seconds")
        attempt_id = sid
    else:
        attempt_id = "A-" + secrets.token_hex(8)  # el id siempre lo pone el servidor
    t = max(1, client_time)
    if server_time is not None and t < server_time - TIME_SLACK:
        t = server_time  # el cliente dice que tardó menos de lo que vio el servidor: vale el servidor
    date = date or now_iso()

    # 3) Puntaje calculado aquí, con la misma función que se usó para entrenar.
    if completed:
        r = compute_result(level, rp.hits, rp.misses, rp.movements, t, rp.hints, True)
        score, accuracy, approved = r["score"], r["accuracy"], r["approved"]
        result = "Aprobado" if approved else "No aprobado"
    else:
        score, accuracy, approved, result = 0, 0, False, "Incompleto"
    n_prev = db().execute("SELECT COUNT(*) FROM attempts WHERE user_id=? AND level_id=?",
                          (g.user["id"], level_id)).fetchone()[0]

    db().execute("""INSERT INTO attempts
        (attempt_id,user_id,game_id,level_id,attempt_number,date,score,accuracy,time,movements,errors,
         hints_used,completed,approved,result,hits,misses,events,verified,client_time,server_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (attempt_id, g.user["id"], level.game_id, level_id, n_prev + 1, date, score, accuracy, t,
                  rp.movements, rp.errors, rp.hints, int(completed), int(approved), result,
                  rp.hits, rp.misses, json.dumps(body.get("events"), separators=(",", ":"))[:200_000],
                  int(verified), client_time, server_time))
    if session is not None:
        db().execute("UPDATE attempt_sessions SET finished=1 WHERE attempt_id=?", (attempt_id,))
    db().commit()

    saved = row_to_attempt(db().execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone())
    reco = None
    if completed:
        history = user_attempts(g.user["id"])  # ya incluye el intento recién guardado
        reco = recommend(MODEL, history)  # <-- el mismo SVM entrenado, ejecutándose en el servidor
        db().execute("""INSERT INTO recommendations
            (user_id,attempt_id,level_id,action,prediction,area,game_id,decision,reason,date)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                     (g.user["id"], attempt_id, reco["level_id"], reco["action"], reco["prediction"],
                      reco["area"], reco["game_id"], float(reco["decision"]), reco["reason"], now_iso()))
        db().commit()
    return jsonify({"attempt": saved, "recommendation": reco})


@app.get("/api/recommendation")
@require_auth
def latest_recommendation():
    row = db().execute("SELECT * FROM recommendations WHERE user_id=? ORDER BY id DESC LIMIT 1",
                       (g.user["id"],)).fetchone()
    if not row:
        return jsonify({"recommendation": None})
    return jsonify({"recommendation": {
        "level_id": row["level_id"], "action": row["action"], "prediction": row["prediction"],
        "area": row["area"], "game_id": row["game_id"], "decision": row["decision"], "reason": row["reason"],
    }})


# ---------- Docente: etiquetas para el aprendizaje supervisado ----------
def _all_labels() -> list[dict]:
    return [dict(r) for r in db().execute(
        "SELECT id, student_id, level_id, label, note, teacher_id, assessed_at, created_at "
        "FROM teacher_labels ORDER BY id")]


@app.get("/api/teacher/students")
@require_teacher
def teacher_students():
    """Alumnos con un resumen por nivel (intentos, mejor puntaje) y su última etiqueta."""
    students = db().execute("SELECT id, name, email FROM users WHERE role='student' ORDER BY name").fetchall()
    summary: dict[str, dict] = {}
    for r in db().execute("""SELECT user_id, level_id, COUNT(*) AS tries,
                                    SUM(completed) AS done, MAX(CASE WHEN completed THEN score END) AS best,
                                    MAX(date) AS last_date
                             FROM attempts GROUP BY user_id, level_id"""):
        summary.setdefault(r["user_id"], {})[r["level_id"]] = {
            "tries": r["tries"], "done": r["done"] or 0, "best": r["best"], "last_date": r["last_date"]}
    last_label: dict[tuple, dict] = {}
    for lab in _all_labels():
        last_label[(lab["student_id"], lab["level_id"])] = lab   # ordenadas por id: queda la última
    out = []
    for s in students:
        levels = {}
        for lid in LEVEL_ORDER:
            info = dict(summary.get(s["id"], {}).get(lid, {"tries": 0, "done": 0, "best": None, "last_date": None}))
            lab = last_label.get((s["id"], lid))
            info["label"] = None if lab is None else {"id": lab["id"], "label": lab["label"],
                                                      "assessed_at": lab["assessed_at"], "note": lab["note"]}
            levels[lid] = info
        out.append({"id": s["id"], "name": s["name"], "email": s["email"], "levels": levels})
    return jsonify({"students": out, "levels": LEVEL_ORDER})


@app.post("/api/teacher/labels")
@require_teacher
def teacher_add_label():
    """Registra una evaluación del docente: el alumno domina (1) o no (0) el nivel."""
    body = request.get_json(silent=True) or {}
    student_id, level_id = body.get("student_id"), body.get("level_id")
    label = body.get("label")
    if level_id not in LEVELS or label not in (0, 1, True, False):
        return error("invalid_input", 400)
    st = db().execute("SELECT role FROM users WHERE id=?", (student_id,)).fetchone()
    if not st or st["role"] != "student":
        return error("invalid_student", 400)
    now = datetime.now(timezone.utc)
    when = now
    if body.get("assessed_at"):
        try:
            when = min(parse_date(body["assessed_at"]), now)   # una evaluación no puede ser futura
        except ValueError:
            return error("invalid_date", 400)
    note = str(body.get("note") or "")[:500] or None
    cur = db().execute("""INSERT INTO teacher_labels (student_id,level_id,label,note,teacher_id,assessed_at,created_at)
                          VALUES (?,?,?,?,?,?,?)""",
                       (student_id, level_id, int(label), note, g.user["id"],
                        when.isoformat(timespec="seconds"), now_iso()))
    db().commit()
    return jsonify({"label": {"id": cur.lastrowid, "student_id": student_id, "level_id": level_id,
                              "label": int(label), "assessed_at": when.isoformat(timespec="seconds"), "note": note}})


@app.delete("/api/teacher/labels/<int:label_id>")
@require_teacher
def teacher_delete_label(label_id: int):
    cur = db().execute("DELETE FROM teacher_labels WHERE id=?", (label_id,))
    db().commit()
    return (jsonify({"ok": True}) if cur.rowcount else error("not_found", 404))


@app.get("/api/teacher/labels")
@require_teacher
def teacher_list_labels():
    return jsonify({"labels": _all_labels()})


@app.get("/api/teacher/dataset.csv")
@require_teacher
def teacher_dataset():
    """Conjunto de entrenamiento listo (sin nombres ni correos): una fila por evaluación,
    con el intento previo que se usará para entrenar. Se puede pasar a train_real.py --csv."""
    attempts = [dict(r) for r in db().execute(
        "SELECT attempt_id,user_id,level_id,date,score,accuracy,time,movements,errors,hints_used,"
        "completed,verified FROM attempts")]
    examples, _ = build_examples(attempts, latest_labels(_all_labels()), verified_only=False)
    buf = io.StringIO()
    write_csv(buf, examples)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=cgame_dataset.csv"})


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5057))
    print(f"[cgame] SVM cargado: {MODEL.meta}")
    print(f"[cgame] Base de datos: {DB_PATH}")
    print(f"[cgame] Escuchando en http://0.0.0.0:{port}/api  (health: /api/health)")
    app.run(host="0.0.0.0", port=port, debug=False)
