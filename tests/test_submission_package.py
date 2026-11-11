"""Execute the actual packaged CSV entry point against a local model-protocol stub."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import build_submission
from test_adaptive import TEXT


class PackagedPipelineTest(unittest.TestCase):
    def test_csv_runs_single_pass_irac_from_packaged_root_modules(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages = request["messages"]
                payload = json.loads(messages[1]["content"])
                source = next(item for item in payload["sources"] if "7 рабочих дней" in item["text"])
                content = ("I — Вопрос\nКогда оплачивать?\nR — Нормы\n"
                           f"Оплата в течение 7 рабочих дней после приёмки [{source['id']}].\n"
                           f"A — Применение\nСрок считают после приёмки [{source['id']}].\n"
                           f"C — Вывод\nОплатить в течение 7 рабочих дней [{source['id']}].")
                calls.append(self.path)
                data = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                   "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "submission.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in build_submission.FILES:
                    self.assertEqual(Path(name).name, name)
                    bundle.write(build_submission.ROOT / name, name)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            data = root / "data"
            data.mkdir()
            with zipfile.ZipFile(data / "kb.zip", "w") as bundle:
                bundle.writestr("Договор М-42-2026.txt", TEXT)
            (data / "questions.csv").write_text('id,question\n7,"Когда оплачивать товар?"\n9,"Срок оплаты?"\n', encoding="utf-8")
            env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                   "LLM_API_KEY": "synthetic-test-key", "LLM_MODEL": "stub", "DATA_DIR": str(data),
                   "USE_EMBEDDINGS": "0", "RERANKER_MODEL": "", "MAX_WORKERS": "2",
                   "PIPELINE_MODE": "single_pass", "HTTP_TRUST_ENV": "0"}
            process = subprocess.run([sys.executable, "-X", "utf8", "solution.py"], cwd=root,
                                     env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            with (root / "submission.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["id"] for row in rows], ["7", "9"])
            for row in rows:
                self.assertIn("7 рабочих дней", row["answer"])
                for heading in ("I — Вопрос", "R — Нормы", "A — Применение", "C — Вывод"):
                    self.assertEqual(row["answer"].count(heading), 1)
            self.assertEqual(calls, ["/v1/chat/completions"] * 2)
            metrics = [json.loads(line) for line in (root / "submission.metrics.jsonl").read_text().splitlines()]
            rows = [item for item in metrics if "id" in item]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(item["pipeline_calls"] == 1 for item in rows))
            self.assertTrue(all(item["prompt_tokens"] == 11 for item in rows))


if __name__ == "__main__":
    unittest.main()
