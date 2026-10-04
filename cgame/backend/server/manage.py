"""Tareas de administración del servidor de CGAME (se corren en la consola del servidor).

    python manage.py make-teacher correo@escuela.edu   # da rol docente a una cuenta ya registrada
    python manage.py make-student correo@escuela.edu   # se lo quita
    python manage.py users                             # lista cuentas, rol y cantidad de intentos
    python manage.py stats                             # cuántos datos hay para entrenar

Usa la misma base que el servidor (variable CGAME_DB_PATH o backend/server/cgame.db).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CGAME_SECRET_KEY", "cli-no-firma-tokens")  # el CLI no emite tokens
from app import DB_PATH, init_db  # noqa: E402
from recommender.dataset import build_examples, load_from_db  # noqa: E402


def set_role(email: str, role: str) -> int:
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        cur = conn.execute("UPDATE users SET role=? WHERE email=?", (role, email.strip().lower()))
    if not cur.rowcount:
        print(f"No existe una cuenta con el correo {email}. Primero regístrala en el juego.")
        return 1
    print(f"{email} ahora tiene rol '{role}'. Debe cerrar sesión y volver a entrar en el juego.")
    return 0


def list_users() -> int:
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        rows = conn.execute("""SELECT u.email, u.name, u.role, COUNT(a.attempt_id)
                               FROM users u LEFT JOIN attempts a ON a.user_id = u.id
                               GROUP BY u.id ORDER BY u.role, u.name""").fetchall()
    for email, name, role, n in rows:
        print(f"{role:8} {n:4} intentos  {name}  <{email}>")
    return 0


def stats() -> int:
    attempts, labels = load_from_db(DB_PATH)
    students = {a["user_id"] for a in attempts}
    verified = [a for a in attempts if a["completed"] and a["verified"]]
    ex, skipped = build_examples(attempts, labels)
    print(f"Intentos: {len(attempts)} ({len(verified)} completados y verificados) de {len(students)} alumnos")
    print(f"Evaluaciones del docente: {len(labels)}")
    print(f"Ejemplos listos para entrenar: {len(ex)} de {len({e['student_id'] for e in ex})} alumnos "
          f"(aprendió: {sum(e['label'] for e in ex)}, no aprendió: {sum(1 - e['label'] for e in ex)})")
    if skipped["sin_intento_previo"]:
        print(f"  {skipped['sin_intento_previo']} evaluaciones sin un intento verificado previo en ese nivel")
    return 0


def main(argv: list[str]) -> int:
    init_db()
    if len(argv) == 2 and argv[0] in ("make-teacher", "make-student"):
        return set_role(argv[1], "teacher" if argv[0] == "make-teacher" else "student")
    if argv == ["users"]:
        return list_users()
    if argv == ["stats"]:
        return stats()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
