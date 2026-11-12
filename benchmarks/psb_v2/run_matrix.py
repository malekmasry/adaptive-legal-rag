"""Reproduce actual solution.py, not a direct import that bypasses ZIP settings.

freeze is offline. run requires an explicit service config JSON outside the repo.
The benchmark gold, judge, anchor scores and rubric NEVER enter the run directory.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ("solution.py", "legal_core.py", "adaptive_pipeline.py", "hallugraph.py",
                 "rag_safety.py", "rag_telemetry.py", "single_pass.py", "submission_runner.py",
                 "answer_review.py", "evidence_coverage.py", "requirements.txt")
ENV_NAMES = set("LLM_BASE_URL LLM_API_KEY LLM_MODEL EMB_BASE_URL EMB_API_KEY EMB_MODEL "
                "RERANKER_BASE_URL RERANKER_API_KEY RERANKER_MODEL TOP_K RETRIEVAL_CANDIDATES "
                "CONTEXT_MAX_CHARS CONTEXT_MAX_SOURCES MAX_TOKENS TEMPERATURE MAX_WORKERS "
                "PIPELINE_TIMEOUT_SECONDS RETRIEVAL_TIMEOUT_SECONDS INDEX_TIMEOUT_SECONDS "
                "API_TIMEOUT_SECONDS MAX_API_ATTEMPTS BATCH_TIMEOUT_SECONDS BATCH_WRITE_RESERVE_SECONDS "
                "HTTP_TRUST_ENV".split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def anchors():
    return json.loads((ROOT / "anchors.json").read_text())["versions"]


def freeze(repo, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    versions = []
    for anchor in anchors():
        commit = subprocess.check_output(["git", "rev-parse", anchor["commit"] + "^{commit}"], cwd=repo, text=True).strip()
        files = subprocess.check_output(["git", "ls-tree", "--name-only", commit], cwd=repo, text=True).splitlines()
        package = target / (anchor["id"] + ".zip")
        included = []
        with ZipFile(package, "w", ZIP_DEFLATED) as archive:
            for name in RUNTIME_FILES:
                if name in files:
                    content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
                    if name.endswith(".py"):
                        ast.parse(content, filename=name)
                    archive.writestr(name, content)
                    included.append(name)
        if "solution.py" not in included:
            raise ValueError("Missing entry point")
        versions.append({**anchor, "package": package.name, "package_sha256": digest(package),
                         "files": included, "package_origin": "reconstructed_from_git_commit",
                         "historical_zip_bytes_verified": False})
    write_json(target / "manifest.json", {"versions": versions})
    return versions


def export_questions(destination, split, ids=None):
    rows = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text().splitlines()]
    chosen = [row for row in rows if row["split"] == split and (ids is None or row["id"] in ids)]
    if not chosen or ids is not None and set(ids) != {row["id"] for row in chosen}:
        raise ValueError("Unknown IDs or IDs outside selected split")
    with Path(destination).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "question"])
        writer.writerows((row["id"], row["question"]) for row in chosen)
    return [row["id"] for row in chosen]


def run(args):
    frozen = json.loads((args.frozen / "manifest.json").read_text())
    anchor = next(row for row in frozen["versions"] if row["id"] == args.version)
    package = args.frozen / anchor["package"]
    if digest(package) != anchor["package_sha256"]:
        raise ValueError("Frozen package was modified")
    config = json.loads(args.services.read_text())
    if not isinstance(config, dict) or not all(isinstance(v, str) for v in config.values()):
        raise ValueError("Service configuration must be an object of strings")
    unknown = set(config) - ENV_NAMES
    if unknown:
        raise ValueError("Unknown configuration keys: " + ",".join(sorted(unknown)))
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "EMB_MODEL", "RERANKER_MODEL"):
        if not config.get(name):
            raise ValueError("Missing explicit setting: " + name)
    for prefix in ("EMB", "RERANKER"):
        if config.get(prefix + "_BASE_URL") and not config.get(prefix + "_API_KEY"):
            raise ValueError("Separate service requires its own key: " + prefix)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with ZipFile(package) as archive:
        for name in archive.namelist():
            if name not in RUNTIME_FILES or Path(name).name != name:
                raise ValueError("Non-runtime file in frozen package")
        archive.extractall(root)
    data = root / "data"
    data.mkdir()
    (data / "kb.zip").write_bytes(args.kb.read_bytes())
    expected_corpus = json.loads((ROOT / "manifest.json").read_text())["archive_sha256"]
    if digest(data / "kb.zip") != expected_corpus:
        raise ValueError("Knowledge base differs from the benchmark corpus")
    ids = export_questions(data / "questions.csv", args.split, args.ids.split(",") if args.ids else None)
    # Keep system/proxy settings, but never inherit old RAG experiment flags.
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR",
                      "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"}}
    env.update(config)
    env.update(DATA_DIR=str(data), OUTPUT_CSV=str(root / "submission.csv"),
               METRICS_PATH=str(root / "submission.metrics.jsonl"), PIPELINE_MODE="single_pass")
    public = {key: value for key, value in config.items() if not key.endswith("_KEY") and not key.endswith("_BASE_URL")}
    public["endpoint_fingerprints"] = {key: hashlib.sha256(value.encode()).hexdigest()
                                       for key, value in config.items() if key.endswith("_BASE_URL")}
    metadata = {"version_id": anchor["id"], "commit": anchor["commit"], "package_sha256": digest(package),
                "dataset_sha256": digest(ROOT / "cases.jsonl"), "corpus_sha256": expected_corpus,
                "split": args.split, "ids": ids, "settings": public, "entry_point": "solution.py",
                "historical_config_confirmed": anchor["historical_config_confirmed"],
                "service_profile_sha256": hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}
    write_json(root / "run.meta.json", metadata)
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, "solution.py"], cwd=root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        metadata["outer_timeout"] = True
    # Never retain a credential if a different historical version printed it.
    for name, raw in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        text = raw.decode("utf-8", errors="replace")
        for key, value in config.items():
            if key.endswith("_KEY") and value:
                text = text.replace(value, "[REDACTED]")
        (root / name).write_text(text, encoding="utf-8")
    metadata.update(exit_code=process.returncode, elapsed_seconds=round(time.perf_counter() - started, 3))
    metrics = root / "submission.metrics.jsonl"
    events = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
    init = next((row for row in events if row.get("stage") == "initialization"), {})
    rows = [row for row in events if "id" in row]
    dense_ready = init.get("dense_index_ready")
    if dense_ready is None:
        # The 14-point revision predates the explicit indexing flag. An actual
        # dense_used event proves that both the index and query embedding worked.
        dense_ready = any(event.get("dense_used") is True for row in rows for event in row.get("events", []))
    metadata["service_observations"] = {
        "dense_index_ready": dense_ready,
        "rerank_api_calls": sum(call.get("endpoint") == "/rerank" for row in rows for call in row.get("api_calls", [])),
        "rerank_api_successes": sum(call.get("endpoint") == "/rerank" and call.get("http_status") == 200
                                     for row in rows for call in row.get("api_calls", [])),
        "technical_errors": sum(row.get("status") == "technical_error" for row in rows),
        "answer_rows": len(rows)}
    metadata["service_check_passed"] = bool(dense_ready and
        metadata["service_observations"]["rerank_api_successes"] and len(rows) == len(ids) and
        {row['id'] for row in rows} == set(ids))
    if (root / "submission.csv").exists():
        metadata["answers_sha256"] = digest(root / "submission.csv")
    write_json(root / "run.meta.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("version_id", "elapsed_seconds", "exit_code", "service_check_passed")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--version", choices=[row["id"] for row in anchors()], required=True)
    p.add_argument("--services", type=Path, required=True)
    p.add_argument("--kb", type=Path, default=ROOT / "knowledge-base.zip")
    p.add_argument("--split", choices=["development", "holdout", "official_demo"], default="development")
    p.add_argument("--ids")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        versions = freeze(args.repo, args.output)
        print(json.dumps({"frozen": len(versions), "api_calls": 0}))
    else:
        run(args)


if __name__ == "__main__":
    main()
