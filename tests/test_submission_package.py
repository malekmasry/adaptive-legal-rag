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
    def test_dialogue_zip_repeats_without_search_and_isolates_csv_ids(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.loads(request["messages"][1]["content"])
                calls.append(payload)
                source = next(s for s in payload["sources"] if "7 рабочих дней" in s["text"])
                content = ("I — Вопрос\nКогда оплачивать?\nR — Нормы\n"
                           f"Оплата в течение 7 рабочих дней [{source['id']}].\n"
                           f"A — Применение\nСрок после приёмки [{source['id']}].\n"
                           f"C — Вывод\nОплатить в течение 7 рабочих дней [{source['id']}].")
                body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "submission.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in build_submission.FILES:
                    bundle.write(build_submission.ROOT / name, name)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            data = root / "data"
            data.mkdir()
            with zipfile.ZipFile(data / "kb.zip", "w") as bundle:
                bundle.writestr("Договор М-42-2026.txt", TEXT)
            rows = [
                ("repeat", "Пользователь: Срок оплаты?\nАссистент: Предыдущий ответ: семь дней.\nПользователь: Повтори ответ"),
                ("missing", "Повтори ответ"),
                ("reanswer", "Пользователь: Когда оплачивать по М-42/2026?\nАссистент: Уточните.\nПользователь: Ответь на предыдущий вопрос"),
            ]
            with (data / "questions.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "question"])
                writer.writerows(rows)
            env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                   "LLM_API_KEY": "stub", "LLM_MODEL": "stub", "RERANKER_MODEL": "",
                   "RERANKER_BASE_URL": "", "RERANKER_API_KEY": "", "DATA_DIR": str(data),
                   "MAX_WORKERS": "1", "HTTP_TRUST_ENV": "0", "PIPELINE_MODE": "single_pass",
                   "OUTPUT_CSV": "submission.csv", "METRICS_PATH": "submission.metrics.jsonl"}
            run = subprocess.run([sys.executable, "-X", "utf8", "solution.py"], cwd=root, env=env,
                                 capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            with (root / "submission.csv").open(encoding="utf-8", newline="") as stream:
                output = {r["id"]: r["answer"] for r in csv.DictReader(stream)}
            self.assertIn("Предыдущий ответ: семь дней", output["repeat"])
            self.assertIn("Предыдущего ответа", output["missing"])
            self.assertNotIn("семь дней", output["missing"])
            self.assertIn("7 рабочих дней", output["reanswer"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["latest_question"], "Когда оплачивать по М-42/2026?")
            metrics = [json.loads(line) for line in (root / "submission.metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            per_id = {r["id"]: r for r in metrics if "id" in r}
            self.assertEqual(per_id["repeat"]["pipeline_calls"], 0)
            self.assertEqual(per_id["missing"]["pipeline_calls"], 0)
            self.assertEqual(per_id["repeat"]["api_calls"], [])
            self.assertEqual(per_id["reanswer"]["pipeline_calls"], 1)

    def test_conditional_zip_enables_embeddings_and_routes_both_questions(self):
        calls = []
        rerank_counts = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, request))
                if self.path.endswith("/embeddings"):
                    response = {"data": [{"index": i, "embedding": [1, 0]}
                                         for i in range(len(request["input"]))]}
                elif self.path.endswith("/rerank"):
                    query = request["query"]
                    rerank_counts[query] = rerank_counts.get(query, 0) + 1
                    score = 0.1 if "СЛАБЫЙ" in query and rerank_counts[query] == 1 else 0.9
                    response = {"results": [{"index": i, "relevance_score": score}
                                            for i in range(len(request["documents"]))]}
                elif self.path.endswith("/chat/completions"):
                    payload = json.loads(request["messages"][1]["content"])
                    source = next(item for item in payload["sources"] if "7 рабочих дней" in item["text"])
                    response = {"choices": [{"finish_reason": "stop", "message": {"content": (
                        "I — Вопрос\nКогда оплачивать?\nR — Нормы\n"
                        f"Оплата в течение 7 рабочих дней после приёмки [{source['id']}].\n"
                        f"A — Применение\nСрок считают после приёмки [{source['id']}].\n"
                        f"C — Вывод\nОплатить в течение 7 рабочих дней [{source['id']}].")}}]}
                else:
                    self.send_error(404)
                    return
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "submission.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in build_submission.FILES:
                    bundle.write(build_submission.ROOT / name, name)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            data = root / "data"
            data.mkdir()
            with zipfile.ZipFile(data / "kb.zip", "w") as bundle:
                bundle.writestr("Договор М-42-2026.txt", TEXT)
            (data / "questions.csv").write_text(
                'id,question\n1,"СИЛЬНЫЙ: Когда оплачивать товар?"\n2,"СЛАБЫЙ: Срок оплаты?"\n', encoding="utf-8")
            env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                   "LLM_API_KEY": "synthetic-test-key", "LLM_MODEL": "stub", "DATA_DIR": str(data),
                   "USE_EMBEDDINGS": "0", "RETRIEVAL_MODE": "hybrid", "CONDITIONAL_DENSE_THRESHOLD": "0.5",
                   "RERANKER_MODEL": "stub-reranker", "EMB_MODEL": "stub-embeddings",
                   "EMB_BASE_URL": "", "EMB_API_KEY": "", "RERANKER_BASE_URL": "", "RERANKER_API_KEY": "",
                   "MAX_WORKERS": "2", "PIPELINE_MODE": "single_pass", "HTTP_TRUST_ENV": "0",
                   "OUTPUT_CSV": "submission.csv", "METRICS_PATH": "submission.metrics.jsonl"}
            process = subprocess.run([sys.executable, "-X", "utf8", "solution.py"], cwd=root,
                                     env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            with (root / "submission.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["id"] for row in rows], ["1", "2"])
            self.assertTrue(all("7 рабочих дней" in row["answer"] for row in rows))
            metrics = [json.loads(line) for line in (root / "submission.metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(metrics[0]["retrieval_mode"], "conditional")
            self.assertTrue(metrics[0]["dense_index_ready"])
            answers = {row["id"]: row for row in metrics if "id" in row}
            routes = {key: next(event for event in row["events"] if event["stage"] == "retrieval_route")
                      for key, row in answers.items()}
            self.assertEqual(routes["1"]["route"], "lexical")
            self.assertEqual(routes["2"]["route"], "dense")
            self.assertEqual(routes["2"]["rerank_score"], 0.1)
            self.assertEqual(routes["2"]["threshold"], 0.5)
            self.assertTrue(all(row["pipeline_calls"] == 1 for row in answers.values()))
            self.assertEqual([path for path, _ in calls].count("/v1/rerank"), 3)
            self.assertEqual([path for path, _ in calls].count("/v1/chat/completions"), 2)
            embeds = [request for path, request in calls if path == "/v1/embeddings"]
            self.assertEqual(len(embeds), 2)  # One corpus batch, one weak query.
            self.assertEqual(len(embeds[-1]["input"]), 1)
            self.assertIn("СЛАБЫЙ", embeds[-1]["input"][0])

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
