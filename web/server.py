"""Local review UI and JSON API for the v243 RAG runtime."""
import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent / "focused-rag"))

from model_api import API
from pipeline import Pipeline, TECHNICAL
from settings import Settings

LOG = logging.getLogger("legal-rag")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY = 64_000
MAX_QUESTION = 4_000


class InputError(ValueError):
    pass


class Assistant:
    def __init__(self, demo):
        self.demo = demo
        self.examples = json.loads((WEB / "examples.json").read_text(encoding="utf-8"))
        self.pipeline = None
        self.lock = threading.Lock()

    def ask(self, question):
        if self.demo:
            for item in self.examples:
                if question == item["question"]:
                    return {"answer": item["answer"], "status": item["status"]}
            raise InputError("В демо доступны готовые примеры. Для своего вопроса включите режим с моделями.")

        if not self.lock.acquire(blocking=False):
            raise BlockingIOError
        try:
            if self.pipeline is None:
                settings = Settings()
                if not (settings.base_url and settings.api_key and settings.model):
                    raise InputError("Укажите LLM_BASE_URL, LLM_API_KEY и LLM_MODEL в настройках сервера.")
                api = API(settings)
                try:
                    candidate = Pipeline(settings, api)
                    candidate.initialize()
                except FileNotFoundError:
                    api.close()
                    raise InputError("Поместите knowledge-base.zip в каталог данных сервера.") from None
                except Exception:
                    api.close()
                    raise
                self.pipeline = candidate
            self.pipeline.api.deadline = time.monotonic() + self.pipeline.s.deadline_seconds
            result = self.pipeline.ask(question)
            return {key: result[key] for key in ("answer", "status")}
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def send_payload(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_payload(status, "application/json; charset=utf-8", data)

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/healthz":
            self.send_json(200, {"status": "ok"})
        elif self.path == "/api/examples":
            self.send_json(200, {"demo": self.server.assistant.demo, "examples": [
                {"label": e["label"], "question": e["question"]} for e in self.server.assistant.examples
            ]})
        elif self.path in ASSETS:
            filename, content_type = ASSETS[self.path]
            self.send_payload(200, content_type, (WEB / filename).read_bytes())
        else:
            self.send_json(404, {"error": "Страница не найдена."})

    def do_POST(self):
        if self.path != "/api/ask":
            self.send_json(404, {"error": "Запрос не найден."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "Ожидается JSON."})
            return
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"error": "Некорректная длина запроса."})
                return
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "Запрос пуст или превышает допустимый размер."})
                return
            data = json.loads(self.rfile.read(length))
            question = data.get("question") if isinstance(data, dict) else None
            if not isinstance(question, str) or not (0 < len(question.strip()) <= MAX_QUESTION):
                raise InputError(f"Введите вопрос длиной до {MAX_QUESTION} символов.")
            self.send_json(200, self.server.assistant.ask(question.strip()))
        except InputError as exc:
            self.send_json(422, {"error": str(exc)})
        except (json.JSONDecodeError, UnicodeDecodeError, TimeoutError, RecursionError):
            self.send_json(400, {"error": "Некорректный запрос."})
        except BlockingIOError:
            self.send_json(503, {"error": "Ассистент отвечает на другой вопрос. Попробуйте чуть позже."})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            LOG.warning("Answer failed (%s)", type(exc).__name__)
            self.send_json(503, {"error": TECHNICAL})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8080")))
    args = parser.parse_args()
    mode = os.getenv("APP_MODE", "demo")
    if mode not in {"demo", "live"}:
        parser.error("APP_MODE must be demo or live")
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        server.assistant = Assistant(demo=mode == "demo")
        LOG.info("Listening on port %d (%s)", args.port, mode)
        server.serve_forever()


if __name__ == "__main__":
    main()
