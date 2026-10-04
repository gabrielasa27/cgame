# Servidor de CGAME (API real, con el SVM entrenado corriendo del lado del servidor)

Este servidor (Flask) expone exactamente los endpoints que `client/index.html` ya sabe
consumir cuando lo abres con `?api=https://tu-servidor/api`. No reimplementa el SVM:
importa `backend/recommender/` tal cual (`compute_result`, `recommend`, `LinearSVM`) y
usa los mismos coeficientes de `backend/data/svm_model.json` que ya viste en el juego.

No puedo desplegarlo por ti (este entorno de chat no tiene salida a internet), pero
aquí tienes el código ya probado — funcionó de punta a punta contra el juego real en
pruebas automatizadas — y los pasos para ponerlo en línea tú mismo. Elige una opción.

## 0. Probarlo en tu computadora (2 minutos)

```bash
cd cgame/backend/server
pip install -r requirements.txt
export CGAME_SECRET_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
python app.py
```

Verás `Escuchando en http://0.0.0.0:5057/api`. Ahora abre `client/index.html` (el
archivo local, no el publicado) agregando el parámetro:

```
client/index.html?api=http://127.0.0.1:5057/api
```

Regístrate y juega: los datos ya se están guardando en `cgame.db` (SQLite) y la
recomendación que ves viene de este proceso Python, no del navegador.

**Importante:** para que el juego *publicado* en claude.ai use este servidor, el
servidor tiene que tener una URL pública (https) — `http://127.0.0.1` solo funciona
si abres el juego desde la misma computadora donde corre el servidor. Para que lo usen
estudiantes desde otros dispositivos, necesitas una de las opciones de abajo.

## 1. Opción más simple: Render.com (gratis)

1. Sube la carpeta `cgame/` a un repositorio de GitHub.
2. En [render.com](https://render.com), crea cuenta → **New** → **Blueprint** → conecta
   el repositorio. Render detecta `backend/server/render.yaml` automáticamente y arma el
   servicio (usa el `Dockerfile` incluido).
3. Espera el build (unos 2-3 minutos). Render te da una URL como
   `https://cgame-api-xxxx.onrender.com`.
4. Prueba que responde: abre `https://cgame-api-xxxx.onrender.com/api/health` en el
   navegador — debe devolver un JSON con `"ok": true`.
5. Abre el juego publicado agregando `?api=`:
   `https://claude.ai/artifact/CQH6y6BguTFkdFycKbdq4g?api=https://cgame-api-xxxx.onrender.com/api`

   (En el plan gratuito, Render "duerme" el servicio tras 15 minutos sin uso; la
   primera visita tras eso tarda ~30 s en responder mientras despierta. Para un salón de
   clases en vivo, conviene el plan pago o cualquiera de las opciones de abajo.)

## 2. Railway.app / Fly.io (gratis con límites, no duermen igual)

Ambos leen el mismo `Dockerfile`:

- **Railway**: New Project → Deploy from GitHub repo → selecciona `cgame` → cuando
  pregunte el Dockerfile, apunta a `backend/server/Dockerfile` con contexto en la raíz
  del repo → agrega la variable de entorno `CGAME_SECRET_KEY` (genera una con
  `python3 -c "import secrets;print(secrets.token_hex(32))"`) → agrega un volumen
  persistente montado en `/var/data` y pon `CGAME_DB_PATH=/var/data/cgame.db` si quieres
  que los datos sobrevivan a los redeploys.
- **Fly.io**: `fly launch` dentro de `cgame/` (detecta el Dockerfile), `fly volumes
  create cgame_data --size 1`, monta el volumen en `/var/data`, define
  `CGAME_SECRET_KEY` con `fly secrets set`, y `fly deploy`.

## 3. Un VPS propio (DigitalOcean, un servidor de la escuela, etc.)

```bash
git clone <tu-repo> && cd cgame/backend/server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export CGAME_SECRET_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
export CGAME_DB_PATH=/var/lib/cgame/cgame.db   # crea esa carpeta antes
gunicorn app:app --bind 127.0.0.1:5057 --workers 2 --threads 4 --daemon
```

Pon Nginx o Caddy delante para servir HTTPS (el navegador exige https para que
`fetch()` funcione desde una página en claude.ai). Con Caddy es una sola línea en el
`Caddyfile`:

```
tu-dominio.com {
    reverse_proxy 127.0.0.1:5057
}
```

Caddy obtiene el certificado HTTPS automáticamente.

## Variables de entorno

| Variable | Para qué sirve | Por defecto |
|---|---|---|
| `CGAME_SECRET_KEY` | Firma los tokens de sesión. **Defínela siempre en producción** — si no, se genera una al azar en cada arranque y las sesiones se invalidan al reiniciar. | aleatoria por arranque |
| `CGAME_DB_PATH` | Dónde guardar la base SQLite. Ponla en un disco persistente si tu plataforma reinicia el contenedor sin conservar el disco. | `backend/server/cgame.db` |
| `CGAME_CORS_ORIGIN` | Qué origen puede llamar a la API. `*` funciona para probar; para producción puedes restringirlo a `https://claude.ai`. | `*` |
| `PORT` | Puerto donde escucha (varias plataformas lo definen automáticamente). | `5057` |

## Qué hace y qué no hace este servidor

- Endpoints del juego (todos bajo `/api`): `POST /register`, `POST /login`, `GET /me`,
  `GET /attempts`, `POST /attempts/start`, `POST /attempts`, `GET /recommendation`,
  `GET /model`, `GET /health`.
- Endpoints del docente (requieren rol `teacher`): `GET /teacher/students`,
  `POST /teacher/labels`, `GET /teacher/labels`, `DELETE /teacher/labels/<id>`,
  `GET /teacher/dataset.csv`.
- Cada `POST /attempts` completado llama a `recommend(MODEL, historial)` — la misma
  función de `backend/recommender/recommend.py`, con el modelo entrenado — y guarda la
  recomendación para que `GET /recommendation` la devuelva.
- Guarda todo en SQLite (`users`, `attempts`, `recommendations`, `attempt_sessions`,
  `teacher_labels`). Al arrancar, migra sola una base de la versión anterior: solo
  agrega tablas y columnas, no borra nada.
- **El puntaje lo calcula el servidor.** El juego envía los eventos crudos del intento
  (cada flecha, cada "Ejecutar", cada pista, con su segundo). El servidor los vuelve a
  jugar sobre el mapa real (`recommender/replay.py`) y calcula aciertos, fallos,
  movimientos, errores, pistas y puntaje con `recommender/scoring.py`. Lo que el cliente
  diga sobre `score`, `accuracy`, `errors`, etc. se ignora. Un intento solo cuenta como
  completado si el programa enviado realmente resuelve el nivel; eventos imposibles
  (un "Ejecutar" con un programa distinto del armado, más flechas de las permitidas,
  tiempos que retroceden…) se rechazan con `400 invalid_events`.
- **El servidor cronometra.** Al pulsar "¡Empezar!" el juego pide `POST /attempts/start`
  y recibe el `attempt_id`. Al terminar, si el cliente dice que tardó menos de lo que
  midió el servidor (con 10 s de tolerancia), vale el tiempo del servidor. Esos intentos
  quedan con `verified = 1`. Los de versiones viejas del juego quedan con
  `verified = 0`, y el entrenamiento los ignora salvo que se pida lo contrario.
- **Límite honesto:** nadie puede inventar un puntaje ni un intento imposible, pero un
  estudiante con conocimientos técnicos podría programar un bot que juegue bien por él.
  Eso no se puede impedir desde el servidor.
- No implementa recuperación de contraseña ni verificación de correo.

## Aprendizaje supervisado con datos reales (paso a paso)

Hasta ahora el SVM se entrenó con casos simulados (`recommender/simulate.py`), es decir,
con etiquetas inventadas por el programa. Para que aprenda de alumnos reales hacen falta
etiquetas reales: **la evaluación del docente, hecha fuera del juego**.

1. **Actualizar el servidor.** Sube el código nuevo (en PythonAnywhere: consola Bash →
   `git pull` en la carpeta del proyecto) y recarga la app (pestaña **Web → Reload**).
   La base existente se migra sola al arrancar.
2. **Crear la cuenta docente.** El docente se registra en el juego como cualquier
   usuario. Luego, en la consola del servidor:
   ```bash
   cd backend/server
   python manage.py make-teacher correo-del-docente@escuela.edu
   ```
   Al volver a entrar, el menú muestra **"Panel docente: evaluar alumnos"**.
3. **Los alumnos juegan con el enlace que incluye `?api=`.** Sin `?api=` los datos quedan
   solo en el navegador y no llegan a la base.
4. **El docente evalúa.** Tras una actividad en el aula (en papel, oral, observación),
   abre el panel, elige la fecha de esa evaluación y marca, por alumno y nivel,
   ✔ *aprendió* o ✘ *no aprendió*. Cada marca se asocia al último intento del alumno en
   ese nivel **anterior** a la fecha. Importante: la marca debe salir de la evaluación,
   no del puntaje del juego; si no, el modelo solo aprende a copiar el puntaje.
5. **Ver si ya alcanza.** `python manage.py stats` muestra cuántos intentos verificados y
   evaluaciones hay. El script pide, por defecto, al menos 8 alumnos evaluados y 5
   ejemplos de cada clase. Cuantos más, mejor: con 20–30 alumnos los números empiezan a
   ser estables.
6. **Entrenar y comparar** (desde `backend/`):
   ```bash
   python -m recommender.train_real --db server/cgame.db
   ```
   Separa los alumnos (no los intentos) en entrenamiento y prueba, de modo que ningún
   alumno aparece en los dos lados. Elige el hiperparámetro C con validación cruzada
   también agrupada por alumno, y muestra una tabla con exactitud, exactitud
   balanceada, precisión, recall y F1, calculadas con los alumnos de prueba, para:
   - el SVM nuevo, entrenado con datos reales;
   - lo que hace hoy el juego (SVM simulado + puntaje ≥ 60);
   - el SVM simulado solo;
   - la regla "puntaje ≥ 60" sola.

   Guarda el modelo en `data/svm_model_real.json` y el informe en
   `data/real_training_report.json`, **sin tocar el modelo del juego**.
7. **Instalarlo, solo si es mejor:**
   ```bash
   python -m recommender.train_real --db server/cgame.db --install
   ```
   Solo reemplaza `data/svm_model.json` si el modelo nuevo supera al actual en exactitud
   balanceada con los alumnos de prueba (o si se agrega `--force`). Antes guarda una
   copia del modelo simulado en `data/svm_model.simulado.json`. Después hay que
   recargar la app web. El juego en modo servidor toma los coeficientes del servidor
   (`GET /model`), así que no hace falta volver a publicar el HTML.

### Probar con una clase de ejemplo

Para ver el panel lleno y probar el entrenamiento antes de tener alumnos reales:

```bash
python manage.py demo            # 24 alumnos ficticios juegan a través del servidor + evaluaciones
python manage.py stats
cd .. && python -m recommender.train_real --db "$CGAME_DB_PATH"    # sin --install
python server/manage.py demo --borrar   # antes de usarlo con alumnos reales
```

Las cuentas de ejemplo terminan en `@demo.cgame` (contraseña `demo1234`;
la docente es `docente@demo.cgame`). Sus etiquetas salen de una "habilidad" oculta
inventada, así que los números del entrenamiento solo muestran cómo funciona el proceso,
no dicen nada sobre alumnos reales. `demo --borrar` elimina solo esas cuentas y todo lo
suyo; los datos reales no se tocan.

Otras opciones: `--csv cgame_dataset.csv` entrena desde el CSV que descarga el panel
docente (sin entrar al servidor), `--which all` usa todos los intentos previos a cada
evaluación en lugar de solo el último, y `--include-unverified` incluye intentos sin
cronómetro del servidor.
