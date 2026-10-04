"""Tareas de administración del servidor de CGAME (se corren en la consola del servidor).

    python manage.py make-teacher correo@escuela.edu   # da rol docente a una cuenta ya registrada
    python manage.py make-student correo@escuela.edu   # se lo quita
    python manage.py users                             # lista cuentas, rol y cantidad de intentos
    python manage.py stats                             # cuántos datos hay para entrenar
    python manage.py demo                              # carga una clase de EJEMPLO (cuentas @demo.cgame)
    python manage.py demo --borrar                     # borra todo lo de la clase de ejemplo

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
from recommender.demo import DEMO_DOMAIN, DEMO_PASSWORD, populate  # noqa: E402


def set_role(email: str, role: str, quiet: bool = False) -> int:
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        cur = conn.execute("UPDATE users SET role=? WHERE email=?", (role, email.strip().lower()))
    if not cur.rowcount:
        print(f"No existe una cuenta con el correo {email}. Primero regístrala en el juego.")
        return 1
    if not quiet:
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
    with closing(sqlite3.connect(DB_PATH)) as conn:
        n_demo = len(_demo_ids(conn))
    if n_demo:
        print(f"AVISO: {n_demo} cuentas son de EJEMPLO ({DEMO_DOMAIN}). Bórralas con "
              "'python manage.py demo --borrar' antes de entrenar con datos reales.")
    return 0


def _demo_ids(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT id FROM users WHERE email LIKE ?", ("%" + DEMO_DOMAIN,))]


def demo() -> int:
    from app import app
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if _demo_ids(conn):
            print("La clase de ejemplo ya está cargada. Para regenerarla: python manage.py demo --borrar")
            return 1
    print("Cargando una clase de EJEMPLO (24 alumnos ficticios que juegan a través del servidor)...")
    totals = populate(app.test_client(), lambda email: set_role(email, "teacher", quiet=True))
    print(f"\nListo: {totals['students']} alumnos, {totals['attempts']} intentos verificados, "
          f"{totals['labels']} evaluaciones de la docente de ejemplo.")
    print(f"Todas las cuentas terminan en {DEMO_DOMAIN} (contraseña: {DEMO_PASSWORD}).")
    print("Míralas en el Panel docente. Para borrarlas: python manage.py demo --borrar")
    return 0


def demo_delete() -> int:
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        ids = _demo_ids(conn)
        if not ids:
            print("No hay datos de ejemplo cargados.")
            return 0
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM teacher_labels WHERE student_id IN ({marks}) OR teacher_id IN ({marks})", ids + ids)
        for table in ("recommendations", "attempts", "attempt_sessions"):
            conn.execute(f"DELETE FROM {table} WHERE user_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM users WHERE id IN ({marks})", ids)
    print(f"Borradas {len(ids)} cuentas de ejemplo con sus intentos y evaluaciones. Los datos reales no se tocaron.")
    return 0


def main(argv: list[str]) -> int:
    init_db()
    if len(argv) == 2 and argv[0] in ("make-teacher", "make-student"):
        return set_role(argv[1], "teacher" if argv[0] == "make-teacher" else "student")
    if argv == ["users"]:
        return list_users()
    if argv == ["stats"]:
        return stats()
    if argv == ["demo"]:
        return demo()
    if argv in (["demo", "--borrar"], ["demo", "--delete"]):
        return demo_delete()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
