"""La clase de ejemplo se carga a través de la API y se borra sin tocar datos reales."""
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "server"


def manage(db, *args):
    env = {**os.environ, "CGAME_DB_PATH": db, "CGAME_SECRET_KEY": "s" * 64}
    return subprocess.run([sys.executable, "manage.py", *args], cwd=SERVER, env=env, capture_output=True, text=True)


class DemoTest(unittest.TestCase):
    def test_load_and_delete_keeps_real_users(self):
        db = os.path.join(tempfile.mkdtemp(), "d.db")
        manage(db, "users")  # crea las tablas
        with closing(sqlite3.connect(db)) as c, c:
            c.execute("INSERT INTO users (id,name,email,salt,hash,created_at,role) "
                      "VALUES ('U-real','Real','real@escuela.edu','s','h','2026-01-01T00:00:00+00:00','student')")
        r = manage(db, "demo")
        self.assertEqual(r.returncode, 0, r.stderr)
        with closing(sqlite3.connect(db)) as c:
            self.assertGreater(c.execute("SELECT COUNT(*) FROM attempts WHERE verified=1").fetchone()[0], 100)
            self.assertGreater(c.execute("SELECT COUNT(*) FROM teacher_labels").fetchone()[0], 50)
        self.assertEqual(manage(db, "demo").returncode, 1)          # no se carga dos veces
        self.assertEqual(manage(db, "demo", "--borrar").returncode, 0)
        with closing(sqlite3.connect(db)) as c:
            self.assertEqual(c.execute("SELECT email FROM users").fetchall(), [("real@escuela.edu",)])
            for t in ("attempts", "teacher_labels", "recommendations", "attempt_sessions"):
                self.assertEqual(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], 0, t)


if __name__ == "__main__":
    unittest.main()
