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

Nota de seguridad: este es un prototipo académico. El puntaje de cada intento se
calcula en el navegador y se envía ya calculado (no se reenvían los eventos crudos),
así que un cliente malicioso podría falsificarlo. Para un despliegue real con
consecuencias académicas, habría que reenviar los eventos (aciertos/fallos, no solo
el resultado) y recalcular `compute_result` en el servidor antes de guardar.
"""
from __future__ import annotations

import hashlib
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
from flask import Flask, g, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/  -> import recommender
from recommender.levels import LEVELS  # noqa: E402
from recommender.model import LinearSVM  # noqa: E402
from recommender.recommend import recommend  # noqa: E402
from recommender.scoring import PASS_SCORE  # noqa: E402

SECRET_KEY = os.environ.get("CGAME_SECRET_KEY") or secrets.token_hex(32)
DB_PATH = os.environ.get("CGAME_DB_PATH", str(Path(__file__).resolve().parent / "cgame.db"))
CORS_ORIGIN = os.environ.get("CGAME_CORS_ORIGIN", "*")
TOKEN_DAYS = 30
ATTEMPT_FIELDS = ["attempt_id", "user_id", "game_id", "level_id", "attempt_number", "date",
                  "score", "accuracy", "time", "movements", "errors", "hints_used",
                  "completed", "approved", "result"]

if not os.environ.get("CGAME_SECRET_KEY"):
    print("[cgame] AVISO: CGAME_SECRET_KEY no está definida; se generó una clave temporal "
          "que cambiará cada reinicio (los tokens dejarán de servir). Defínela en producción.",
          file=sys.stderr)

app = Flask(__name__)
MODEL = LinearSVM.load()  # mismos coeficientes que ve el juego (svm_model.json)


# ---------- Base de datos ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
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
        CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, date);
        """)
        conn.commit()


# ---------- Utilidades ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def make_token(user_id: str) -> str:
    payload = {"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + TOKEN_DAYS * 86400}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def error(code: str, status: int):
    return jsonify({"error": code}), status


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


def public_user(row) -> dict:
    return {"id": row["id"], "name": row["name"], "email": row["email"], "created_at": row["created_at"]}


def row_to_attempt(row) -> dict:
    d = {k: row[k] for k in row.keys()}
    d["completed"] = bool(d["completed"])
    d["approved"] = bool(d["approved"])
    return d


def user_attempts(user_id: str) -> list[dict]:
    rows = db().execute("SELECT * FROM attempts WHERE user_id=? ORDER BY date ASC", (user_id,)).fetchall()
    return [row_to_attempt(r) for r in rows]


# ---------- CORS ----------
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Vary"] = "Origin"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return "", 204


# ---------- Rutas ----------
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL.meta, "levels": len(LEVELS)})


@app.post("/api/register")
def register():
    body = request.get_json(silent=True) or {}
    name, email, password = (body.get("name") or "").strip(), (body.get("email") or "").strip().lower(), body.get("password") or ""
    if not name or "@" not in email or len(password) < 6:
        return error("invalid_input", 400)
    if db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        return error("email_taken", 409)
    salt = secrets.token_hex(16)
    user_id = "U-" + secrets.token_hex(8)
    row = (user_id, name, email, salt, hash_password(password, salt), now_iso())
    db().execute("INSERT INTO users (id,name,email,salt,hash,created_at) VALUES (?,?,?,?,?,?)", row)
    db().commit()
    return jsonify({"user": {"id": user_id, "name": name, "email": email, "created_at": row[5]}})


@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    email, password = (body.get("email") or "").strip().lower(), body.get("password") or ""
    urow = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not urow or hash_password(password, urow["salt"]) != urow["hash"]:
        return error("bad_credentials", 401)
    return jsonify({"user": public_user(urow), "token": make_token(urow["id"])})


@app.get("/api/me")
@require_auth
def me():
    return jsonify({"user": public_user(g.user)})


@app.get("/api/attempts")
@require_auth
def list_attempts():
    return jsonify({"attempts": user_attempts(g.user["id"])})


@app.post("/api/attempts")
@require_auth
def save_attempt():
    body = request.get_json(silent=True) or {}
    if body.get("level_id") not in LEVELS:
        return error("invalid_level", 400)
    attempt_id = body.get("attempt_id") or ("A-" + secrets.token_hex(8))
    completed = bool(body.get("completed"))
    score = int(body.get("score") or 0)
    approved = bool(completed and score >= PASS_SCORE)
    row = (attempt_id, g.user["id"], body.get("game_id", ""), body["level_id"],
           int(body.get("attempt_number") or 0), body.get("date") or now_iso(),
           score, int(body.get("accuracy") or 0), int(body.get("time") or 0),
           int(body.get("movements") or 0), int(body.get("errors") or 0),
           int(body.get("hints_used") or 0), int(completed), int(approved),
           body.get("result") or ("Aprobado" if approved else ("No aprobado" if completed else "Incompleto")))
    db().execute("""INSERT OR REPLACE INTO attempts
        (attempt_id,user_id,game_id,level_id,attempt_number,date,score,accuracy,time,
         movements,errors,hints_used,completed,approved,result) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
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


init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5057))
    print(f"[cgame] SVM cargado: {MODEL.meta}")
    print(f"[cgame] Base de datos: {DB_PATH}")
    print(f"[cgame] Escuchando en http://0.0.0.0:{port}/api  (health: /api/health)")
    app.run(host="0.0.0.0", port=port, debug=False)
