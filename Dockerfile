FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY focused-rag/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/cache && chown rag:rag /app/cache
COPY focused-rag ./focused-rag
COPY web ./web
USER rag
ENV DATA_DIR=/app/data CACHE_DIR=/app/cache
EXPOSE 8080
CMD ["python", "web/server.py", "--host", "0.0.0.0", "--port", "8080"]
