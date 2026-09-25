FROM python:3.12-slim
WORKDIR /app
COPY backend/recommender ./backend/recommender
COPY backend/data ./backend/data
COPY backend/server ./backend/server
COPY shared ./shared
RUN pip install --no-cache-dir -r backend/server/requirements.txt
ENV PORT=5057
EXPOSE 5057
CMD ["sh", "-c", "gunicorn --chdir backend/server app:app --bind 0.0.0.0:${PORT} --workers 2 --threads 4"]
