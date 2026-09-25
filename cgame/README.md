# CGAME — Videojuego educativo de pensamiento computacional

Prototipo funcional de CGAME: minijuegos de programación lineal para 3.º de primaria,
con un backend de recomendación basado en SVM (scikit-learn) que decide la siguiente
actividad de cada estudiante.

## Cómo jugarlo

Abre **`client/cgame.html`** en cualquier navegador (Chrome, Edge, Firefox). No necesita
instalación ni servidor: funciona como una página normal, guardando los datos del
estudiante en el propio dispositivo (`localStorage`). Hay un botón "Probar con un
estudiante de ejemplo" en la pantalla inicial para ver el juego ya con historial.

## Qué se implementó (según el documento de la investigación)

- **Usuarios**: registro y login (correo + contraseña, con verificación y mensajes de
  error simples). Contraseña resguardada con PBKDF2-SHA256 en el navegador.
- **Menú principal**: Jugar, Seleccionar nivel, Mis resultados, Mi progreso, Ayuda,
  Cerrar sesión.
- **3 minijuegos** de dificultad progresiva (Básico/Intermedio/Avanzado, 3 niveles cada
  uno = 9 niveles en total):
  - *Ruta del Robot*: arma una secuencia de instrucciones para que Cody llegue a la meta.
  - *Estrellas en Camino*: igual, pero primero debe recoger varias estrellas (planificación).
  - *Ordena los Pasos*: ordena tarjetas según la secuencia lógica correcta.
- **Evaluación**: cronómetro, contador de movimientos, errores y ayudas usadas;
  detección de finalización; cálculo de puntaje y aprobado/no aprobado; los intentos
  abandonados se guardan como "Incompleto" y no cuentan como evaluación superada.
- **Resultados**: pantalla con estado, puntaje, % de aciertos, tiempo, movimientos,
  errores, ayudas y mensaje educativo, más una animación que muestra cómo el SVM
  procesa el intento y genera la recomendación.
- **Historial** ("Mis resultados") y **Mi progreso** (con gráfico de evolución).
- **Ayuda contextual**: pistas progresivas que no resuelven el ejercicio completo, con
  aviso automático si el estudiante lleva 30 s sin avanzar. Cada pista resta puntos.
- **Estructura de datos para el SVM**: cada intento guarda exactamente
  `user_id, game_id, level_id, attempt_id, score, accuracy, time, movements, errors,
  hints_used, completed, date` (ver pantalla "Datos para el docente", con exportación
  a CSV).
- **Sistema de recomendación**: tras cada intento completado, el SVM decide si el
  estudiante "aprendió" o "necesita reforzar" y sugiere avanzar, repetir el nivel,
  bajar un nivel (tras 2 fallos seguidos) o repasar el nivel más débil al terminar
  todo el contenido.

## Arquitectura

Cliente-servidor, tal como pide la investigación, con las piezas Python que el
documento asigna al backend (procesamiento de datos y SVM) totalmente implementadas y
probadas, y una capa de cliente que puede operar sola o hablar con un servidor:

```
cgame/
├── shared/levels.json         # Definición única de juegos y niveles (fuente de verdad)
├── backend/
│   ├── recommender/
│   │   ├── levels.py          # Carga niveles + BFS que calcula el nº óptimo de movimientos
│   │   ├── scoring.py         # Cálculo del puntaje / aprobado
│   │   ├── features.py        # Variables de entrada del SVM
│   │   ├── simulate.py        # Generador de casos simulados (entrenamiento y datos de prueba)
│   │   ├── model.py           # Entrenamiento del SVM (scikit-learn) + serialización a JSON
│   │   ├── recommend.py       # Lógica de recomendación (avanzar/reforzar/bajar nivel/repasar)
│   │   └── train.py           # CLI: entrena y exporta modelo + informe + CSV de prueba
│   ├── tests/test_recommender.py   # 13 pruebas unitarias (niveles, puntaje, recomendación)
│   └── data/                  # svm_model.json, training_report.json, sample_attempts.csv
├── client/
│   ├── src/index.template.html     # Código fuente del juego (HTML+CSS+JS, un solo archivo)
│   └── cgame.html                  # Build final: plantilla + niveles + modelo ya inyectados
└── tools/build_client.py      # Inyecta shared/levels.json y el modelo entrenado en el cliente
```

**Por qué un solo archivo HTML en vez de Unity/Django/Firebase.** Unity y Firebase no
se pueden compilar ni desplegar en este entorno de chat, y no hay salida a internet
para instalar Django ni para que tú te conectes a un servidor en vivo. En lugar de
entregar código de esas piezas sin poder ejecutarlo ni probarlo (lo que el prompt pide
evitar explícitamente: "no generes código ficticio ni funciones sin implementar"), se
tradujo la misma arquitectura a piezas que sí se pudieron construir y probar de
extremo a extremo:

- La lógica de **Unity** (el videojuego interactivo) se implementó en HTML5/CSS/JavaScript
  puro: mismo resultado para el estudiante (botones grandes, animaciones, minijuegos),
  ejecutable en cualquier navegador sin instalar nada.
- La lógica de **Python + SVM** (backend/recommender/) es el mismo Python que pediría
  Django, solo que sin el framework web alrededor. Se entrenó con scikit-learn siguiendo
  el diagrama de flujo del documento (entrenar → validar por CV → probar → repetir si no
  se alcanza el 92 % → repetir con más datos).
- La lógica de **Django** (autenticación, guardado de intentos, endpoint de
  recomendación) está definida como una interfaz clara en el cliente
  (`LocalBackend` / `makeRemoteBackend` en `index.template.html`): ambas exponen los
  mismos métodos (`register`, `login`, `saveAttempt`, `latestRecommendation`, etc.). Hoy
  el juego usa `LocalBackend` (guarda en el dispositivo); el día que exista un servidor
  Django real con esos mismos endpoints, basta con abrir el juego como
  `cgame.html?api=https://tu-servidor/api` para que hable con él sin tocar el resto del
  código.
- **Firebase** cumpliría el mismo papel que hoy cumple `localStorage` en el cliente:
  guardar los datos del usuario. Migrar de uno a otro no cambia el resto del sistema,
  porque toda la app pasa por esa misma interfaz de backend.

Se verificó automáticamente (con pruebas de navegador) que la lógica de puntaje y la
lógica de recomendación del cliente JavaScript coinciden exactamente con la del backend
Python en cientos de casos aleatorios, así que ambas piezas están sincronizadas.

## Sobre el SVM

- **Datos**: como en esta etapa no hay resultados reales de estudiantes, se generan
  casos simulados (`recommender/simulate.py`) a partir de una habilidad latente que
  produce aciertos, eficiencia, rapidez, errores y ayudas de forma realista, con una
  etiqueta "aprendió/no aprendió" que depende de esa habilidad y de la dificultad del
  nivel. Esto es exactamente lo que pide la metodología ("a partir de casos previamente
  simulados") y queda documentado en el propio código y en la pantalla "Datos para el
  docente" para que quede claro que aún no son datos reales.
- **Entrenamiento** (`python -m recommender.train` desde `backend/`): sigue el diagrama
  de flujo del documento — entrena un SVM lineal, ajusta el hiperparámetro C por
  validación cruzada (5 pliegues), lo evalúa contra un conjunto de prueba reservado y,
  si no llega a 92 % de precisión y recall, genera más datos simulados y repite (máximo
  5 iteraciones). Con los datos actuales se alcanza el objetivo en la 3.ª iteración
  (4 500 casos): precisión 93.4 %, recall 92.8 %.
  - Importante: el conjunto de prueba nunca se reutiliza para reentrenar (eso sería fuga
    de datos); cada iteración genera datos y una prueba nuevos.
- **Recomendación**: combina la decisión del SVM en el intento más reciente con el
  promedio de los 2 intentos previos en el mismo nivel, y así decide entre avanzar,
  reforzar el mismo nivel, bajar un nivel (dos fallos seguidos) o —si ya se completó
  todo— repasar el nivel con el puntaje más bajo.

## Pruebas realizadas

- 13 pruebas unitarias de Python (niveles solucionables por BFS, puntaje, y las 6 ramas
  de la lógica de recomendación) — todas pasan.
- Pruebas de navegador (Playwright) que comparan 300 casos aleatorios de puntaje y 300
  de recomendación entre el JavaScript del cliente y el Python del backend: 0
  diferencias.
- Flujo completo de un estudiante probado de punta a punta en el navegador: registro,
  validaciones, login con error, menú, elegir juego y nivel, jugar chocando con una
  roca, pedir pistas, completar el nivel, ver el resultado y la recomendación, jugar
  "Ordena los Pasos" fallando y corrigiendo, abandonar un nivel a medias (se guarda como
  "Incompleto"), revisar historial/progreso/datos del docente, recargar la página
  (la sesión persiste) y cerrar sesión — sin errores.

## Si más adelante se agrega un servidor real (Django + Firebase)

**Ya está hecho.** Hay un servidor real en `backend/server/` (Flask, porque Django no
se pudo instalar en este entorno sin conexión, pero expone la misma interfaz que
pedía el diseño original) que envuelve `backend/recommender/` tal cual — mismo SVM,
mismos coeficientes — y guarda todo en una base de datos real (SQLite). Se probó de
punta a punta: registro, login, jugar un nivel y recibir la recomendación calculada
en el servidor, todo contra el juego real. Instrucciones completas de cómo correrlo y
desplegarlo (Render, Railway, Fly.io, un VPS propio) en `backend/server/README.md`.

Una vez que el servidor esté en línea con una URL pública, el juego publicado se
conecta agregando `?api=https://tu-servidor/api` a la URL — sin tocar nada más del
cliente.
