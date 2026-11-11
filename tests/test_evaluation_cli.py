"""The comparison harness exercises direct/adaptive/oracle through an HTTP stub."""
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

from legal_core import LegalAssistant
from test_adaptive import ScriptedProxy, TEXT


class EvaluationCliTest(unittest.TestCase):
    def test_three_modes_and_offline_retrieval_report(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                system, user = [message["content"] for message in data["messages"]]
                calls.append(self.path)
                if "Этап:" in system:
                    text = ScriptedProxy().complete(system, user)
                else:
                    source = next(item for item in json.loads(user)["sources"] if "7 рабочих дней" in item["text"])
                    text = ("I — Вопрос\nКогда платить?\nR — Нормы\n"
                            f"7 рабочих дней после приёмки [{source['id']}].\n"
                            f"A — Применение\nСрок отсчитывается после приёмки [{source['id']}].\n"
                            f"C — Вывод\nОплатите в течение 7 рабочих дней [{source['id']}].")
                response = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                                       "usage": {"prompt_tokens": 100, "completion_tokens": 50}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "kb.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("contract.txt", TEXT)
            engine = LegalAssistant(archive, ScriptedProxy(), use_embeddings=False)
            source = next(source for source in engine.sources if source.clause == "1.1")
            cases = root / "cases.json"
            cases.write_text(json.dumps({"cases": [{"id": "T1", "question": "Когда оплачивать товар?",
                                                    "gold_groups": [[source.id]]}]}), encoding="utf-8")
            report = root / "comparison.json"
            env = {**os.environ, "LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "LLM_API_KEY": "stub",
                   "LLM_MODEL": "stub", "USE_EMBEDDINGS": "0", "RERANKER_MODEL": "", "HTTP_TRUST_ENV": "0"}
            result = subprocess.run([sys.executable, "experiments/compare_pipelines.py", "--archive", str(archive),
                                     "--questions", str(cases), "--output", str(report), "--modes", "single_pass", "adaptive", "oracle"],
                                    cwd=repo, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(report.read_text())
            self.assertEqual([row["diagnostics"]["pipeline_calls"] for row in saved["runs"]], [1, 3, 1])
            self.assertEqual([row["status"] for row in saved["runs"]], ["answered"] * 3)
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(row["manual_correctness"] is None for row in saved["runs"]))
            # Even with a reranker configured, the frozen retrieval evaluator must not use an API.
            env["RERANKER_MODEL"] = "must-not-call"
            offline = root / "retrieval.json"
            result = subprocess.run([sys.executable, "experiments/evaluate_frozen_retrieval.py", "--archive", str(archive),
                                     "--questions", str(cases), "--output", str(offline)],
                                    cwd=repo, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(offline.read_text())["summary"]["context_all"], 1)
            self.assertEqual(len(calls), 5)


if __name__ == "__main__":
    unittest.main()
