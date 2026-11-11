"""Real HTTP/subprocess tests, including a response that trickles until cancelled."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import build_submission
from legal_core import ModelProxy
from test_single_pass import TEXT, direct_answer


class DeadlineTests(unittest.TestCase):
    def server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def test_permanent_http_error_is_not_retried(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                calls.append(1)
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()

        server = self.server(Handler)
        proxy = ModelProxy()
        proxy.base, proxy.key = f"http://127.0.0.1:{server.server_port}", "test"
        with patch.dict(os.environ, {"HTTP_TRUST_ENV": "0", "MAX_API_ATTEMPTS": "3"}):
            with self.assertRaises(RuntimeError):
                proxy.post("/chat/completions", {})
        self.assertEqual(len(calls), 1)

    def test_slow_body_obeys_batch_deadline_and_checkpoint_keeps_finished_answer(self):
        slow_started = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.loads(request["messages"][1]["content"])
                slow = "МЕДЛЕННО" in payload["latest_question"]
                content = direct_answer(payload)
                body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    if slow:
                        slow_started.set()
                        for byte in body:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.05)
                    else:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = self.server(Handler)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in build_submission.FILES:
                (root / name).write_bytes((build_submission.ROOT / name).read_bytes())
            data = root / "data"
            data.mkdir()
            with zipfile.ZipFile(data / "kb.zip", "w") as archive:
                archive.writestr("contract.txt", TEXT)
            with (data / "questions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "question"])
                writer.writerow(["007", 'Пользователь: Компания «Лазурный Архив».\nПользователь: Как направить заявку, и куда?'])
                writer.writerow(["009", "МЕДЛЕННО: Как направить заявку?"])
            env = {**os.environ, "DATA_DIR": str(data), "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                   "LLM_API_KEY": "test", "LLM_MODEL": "stub", "PIPELINE_MODE": "single_pass",
                   "MAX_WORKERS": "2", "USE_EMBEDDINGS": "0", "RERANKER_MODEL": "", "HTTP_TRUST_ENV": "0",
                   "BATCH_TIMEOUT_SECONDS": "2", "BATCH_WRITE_RESERVE_SECONDS": "0", "API_TIMEOUT_SECONDS": "10"}
            started = time.perf_counter()
            process = subprocess.Popen([sys.executable, "solution.py"], cwd=root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(slow_started.wait(2))
                checkpoint_seen = False
                while time.perf_counter() - started < 2 and process.poll() is None:
                    output = root / "submission.csv"
                    if output.exists():
                        with output.open(encoding="utf-8", newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        if len(rows) == 2 and "по электронной почте" in rows[0]["answer"]:
                            checkpoint_seen = True
                            break
                    time.sleep(0.02)
                _, stderr = process.communicate(timeout=4)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertTrue(checkpoint_seen, "Finished answer was not saved while another request was running")
                self.assertLess(time.perf_counter() - started, 4)
                with (root / "submission.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["id"] for row in rows], ["007", "009"])
                self.assertIn("по электронной почте", rows[0]["answer"])
                self.assertTrue(rows[1]["answer"].strip())
                metrics = [json.loads(line) for line in (root / "submission.metrics.jsonl").read_text().splitlines()]
                self.assertEqual({row["id"]: row["status"] for row in metrics if "id" in row},
                                 {"007": "answered", "009": "technical_error"})
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
