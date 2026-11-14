"""Score a contestant against the frozen v4 BM25 reference, never select a winner.

Reuses csv_judge.evaluate unchanged (two answer orders and its existing retries).
All available raw attempts returned by that evaluator are retained, including
failed validations. The legacy evaluator does not retain transport-error bodies
or responses rejected before its attempt logger; this harness does not invent
them. Completed error/review rows are preserved on resume, not silently retried.
"""
from __future__ import annotations
from model_roles import JUDGE_MODEL

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import csv_judge
from benchmark import ROOT, read_answers
from run_matrix import RUNTIME_FILES
from selection import selection_summary

PROTOCOL = "fixed-bm25-common-reference-v5"
REFERENCE_ROLE = "fixed_bm25_reference"
DEFAULT_MODEL = JUDGE_MODEL
EVALUATE_SOURCE_SHA256 = hashlib.sha256(inspect.getsource(csv_judge.evaluate).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_cases(path):
    rows = csv_judge.load_rows(path)
    if not rows:
        raise ValueError("Cases must not be empty")
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "question")):
            raise ValueError("Cases require nonempty string IDs and questions")
        family = row.get("question_family") or row.get("family_id")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Cases require a question family")
    return rows


def _questions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "question"]:
            raise ValueError("Question CSV columns must be exactly id,question")
        result = {}
        for row in reader:
            if set(row) != {"id", "question"} or any(not row.get(k) for k in ("id", "question")):
                raise ValueError("Malformed question CSV row")
            if row["id"] in result:
                raise ValueError("Duplicate question CSV ID")
            result[row["id"]] = row["question"]
    return result


def _manifest_packages(root):
    """Only local frozen/candidate registries establish trusted package identity."""
    frozen = root / "frozen/manifest.json"
    if frozen.exists():
        for entry in json.loads(frozen.read_text(encoding="utf-8")).get("versions", []):
            yield frozen, entry, entry.get("package"), entry.get("package_sha256")
    manifests = sorted((root / "v3/candidates").glob("*/manifest.json"))
    manifests += sorted((root / "v4/candidates").glob("*/manifest.json"))
    manifests += sorted((root / "v4/packages").glob("*.manifest.json"))
    for manifest in manifests:
        entry = json.loads(manifest.read_text(encoding="utf-8"))
        expected = entry.get("package_sha256") or entry.get("runtime_sha256") or entry.get("sha256")
        default = manifest.name.removesuffix(".manifest.json") + ".zip" if manifest.name.endswith(".manifest.json") else "runtime.zip"
        yield manifest, entry, entry.get("package", default), expected


def _verify_package(run_dir, expected, root=ROOT, reference=False):
    manifest=root/'v5/plan.json'; plan=json.loads(manifest.read_text())
    matches=[(name,entry) for name,entry in plan['packages'].items() if entry['sha256']==expected]
    if len(matches)!=1:raise ValueError('Package absent from frozen v5 plan')
    name,entry=matches[0]
    if reference and name!='reference':raise ValueError('Wrong frozen reference')
    package=root/'v5/packages'/entry['package']
    if sha(package)!=expected:raise ValueError('Frozen archive changed')
    with ZipFile(package) as z:
        if len(z.namelist())!=len(entry['files']) or set(z.namelist())!=set(entry['files']):raise ValueError('Invalid package members')
        if any(hashlib.sha256(z.read(n)).hexdigest()!=v for n,v in entry['files'].items()):raise ValueError('Invalid archive bytes')
    if {p.name for p in Path(run_dir).glob('*.py')}!={n for n in entry['files'] if n.endswith('.py')}:raise ValueError('Runtime modules changed')
    if any(sha(Path(run_dir)/n)!=v for n,v in entry['files'].items()):raise ValueError('Runtime file changed')
    return {'package_sha256':expected,'manifest_sha256':sha(manifest),'runtime_file_sha256':entry['files']}


def verify_run(run_dir, rows, cases_sha256, corpus_sha256, *, reference=False, root=ROOT):
    run_dir = Path(run_dir)
    meta_path = run_dir / "run.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("valid_run") is not True or meta.get("all_ids_valid") is not True or meta.get("exit_code") != 0:
        raise ValueError("Generation run is incomplete or invalid")
    if reference and meta.get("evaluation_role") != REFERENCE_ROLE:
        raise ValueError("Reference run lacks the fixed BM25 evaluation role")
    for field, path in (("answers_sha256", run_dir / "submission.csv"),
                        ("questions_sha256", run_dir / "data/questions.csv"),
                        ("corpus_sha256", run_dir / "data/kb.zip")):
        if meta.get(field) != sha(path):
            raise ValueError("Generation artifact hash mismatch: " + field)
    if meta.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Generation corpus differs from judging corpus")
    questions = _questions(run_dir / "data/questions.csv")
    if list(questions) != meta.get("ids"):
        raise ValueError("Question CSV does not match recorded generation IDs")
    answers = read_answers(run_dir / "submission.csv", list(questions))
    if set(answers) != set(questions):
        raise ValueError("Answer CSV does not exactly match generation IDs")
    if any(questions.get(row["id"]) != row["question"] for row in rows):
        raise ValueError("Selected id/question pair differs from generation input")
    same_cases = meta.get("cases_sha256") == cases_sha256
    if not _digest(meta.get("cases_sha256")):
        raise ValueError("Missing original generation cases fingerprint")
    if not same_cases and (reference or any(row.get("split") != "development" for row in rows)):
        raise ValueError("Only exact development subsets may reuse differently hashed generation cases")
    if same_cases and set(questions) != {row["id"] for row in rows}:
        raise ValueError("Exact cases hash has inconsistent generation IDs")
    observations = meta.get("service_observations", {})
    if observations.get("technical_errors") != 0 or observations.get("completed_answers") != len(questions):
        raise ValueError("Generation service observations are incomplete")
    profile = meta.get("service_profile_sha256")
    settings = meta.get("public_settings")
    if not isinstance(settings, dict) or not _digest(profile) or hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest() != profile:
        raise ValueError("Generation service profile hash mismatch")
    provenance = _verify_package(run_dir, meta.get("package_sha256"), root=root, reference=reference)
    provenance.update(run_meta_sha256=sha(meta_path), answers_sha256=meta["answers_sha256"],
                      questions_sha256=meta["questions_sha256"], generation_cases_sha256=meta["cases_sha256"],
                      service_profile_sha256=profile, generation_ids=meta["ids"],
                      selected_ids=[row["id"] for row in rows], reused_development_subset=not same_cases)
    return answers, provenance


def _configuration(services, model):
    cfg = json.loads(Path(services).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or any(not isinstance(cfg.get(k), str) or not cfg[k].strip() for k in ("LLM_BASE_URL", "LLM_API_KEY")):
        raise ValueError("Judge requires private LLM_BASE_URL and LLM_API_KEY settings")
    endpoint = cfg["LLM_BASE_URL"].rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid judge endpoint")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Judge model must not be empty")
    secrets = [value for key, value in cfg.items() if key.endswith("_KEY") and isinstance(value, str) and value]
    return (endpoint, cfg["LLM_API_KEY"], model), secrets


def _redact(value, secrets):
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secrets): _redact(item, secrets) for key, item in value.items()}
    return value


def _evaluate(row, docs, reference, candidate, config, secrets, evaluation_key):
    try:
        result = csv_judge.evaluate(row, docs, reference, candidate, config)
    except Exception as exc:
        # Unexpected failures outside the legacy evaluator must remain visible.
        result = {"status": "judge_error", "error_type": type(exc).__name__, "rounds": [], "attempts": []}
    family = row.get("question_family") or row["family_id"]
    result.update(id=row["id"], question_family=family, family_id=family, evaluation_key=evaluation_key)
    for key in ("category", "question_type", "split"):
        if key in row:
            result[key] = row[key]
    if row.get("family_id") and row["family_id"] != family:
        result["source_family_id"] = row["family_id"]
    result["technical_error"] = result.get("status") == "judge_error"
    result["unresolved"] = result.get("status") != "ok"
    return _redact(result, secrets)


def _validate_result(result, row, evaluation_key):
    family = row.get("question_family") or row["family_id"]
    if result.get("evaluation_key") != evaluation_key or result.get("question_family") != family or result.get("family_id") != family:
        raise ValueError("Resume result provenance mismatch")
    if result.get("status") not in ("ok", "judge_error", "needs_review") or not isinstance(result.get("attempts"), list) or not isinstance(result.get("rounds"), list):
        raise ValueError("Invalid persisted judgment")
    if result["status"] == "ok" and (len(result["rounds"]) != 2 or type(result.get("score")) not in (int, float) or result["score"] not in [i / 8 for i in range(9)]):
        raise ValueError("Successful judgment requires two orders and a valid score")


def run(cases, kb, reference_run, candidate_run, services, output, workers=2, model=DEFAULT_MODEL):
    rows = load_cases(cases)
    cases_sha256, corpus_sha256 = sha(cases), sha(kb)
    reference, reference_meta = verify_run(reference_run, rows, cases_sha256, corpus_sha256, reference=True)
    candidate, candidate_meta = verify_run(candidate_run, rows, cases_sha256, corpus_sha256)
    if candidate_meta["service_profile_sha256"] != reference_meta["service_profile_sha256"]:
        raise ValueError("Reference and candidate generation service profiles differ")
    config, secrets = _configuration(services, model)
    with ZipFile(kb) as archive:
        names = [name for name in archive.namelist() if name.endswith(".txt")]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate corpus document paths")
        docs = {name: archive.read(name).decode("utf-8-sig") for name in names}
    meta = {"protocol": PROTOCOL, "semantic_protocol": csv_judge.PROTOCOL,
            "scope": "common_reference", "ids": [row["id"] for row in rows],
            "model": model, "judge_endpoint_sha256": hashlib.sha256(config[0].encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(csv_judge.PROMPT.encode()).hexdigest(),
            "inputs": {"cases_sha256": cases_sha256, "corpus_sha256": corpus_sha256},
            "reference": reference_meta, "candidate": candidate_meta,
            "implementation": {name: sha(Path(__file__).with_name(name)) for name in
                               ("v5_common_judge.py", "v5_execute.py", "csv_judge.py", "judge.py", "benchmark.py", "selection.py", "run_matrix.py")},
            "evaluate_source_sha256": EVALUATE_SOURCE_SHA256,
            "answer_orders": [["reference", "candidate"], ["candidate", "reference"]],
            "organizer_equivalence": False, "winner_selection": False,
            "attempt_retention": "All attempts exposed by unchanged csv_judge.evaluate; no transport bodies or pre-logger rejected outputs"}
    run_key = object_sha(meta)
    meta["resume_key"] = run_key
    keys = {row["id"]: object_sha({"run": run_key, "case": row,
                                 "reference": reference[row["id"]], "candidate": candidate[row["id"]]}) for row in rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if Path(output).resolve() in {Path(cases).resolve(), Path(kb).resolve(), Path(services).resolve(),
                                 (Path(reference_run) / "submission.csv").resolve(), (Path(candidate_run) / "submission.csv").resolve()}:
        raise ValueError("Output must not overwrite an input")
    meta_path = output.with_suffix(".meta.json")
    if output == meta_path or output == output.with_suffix(".summary.json"):
        raise ValueError("Use a distinct JSONL output filename")
    with output.with_suffix(".lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This judging output is already running") from None
        if meta_path.exists():
            if json.loads(meta_path.read_text(encoding="utf-8")) != meta:
                raise ValueError("Cannot resume with changed cases, runs, reference, model, endpoint or judge source")
        elif output.exists():
            raise ValueError("Judgments without provenance")
        else:
            write_json(meta_path, meta)
        results = csv_judge.load_rows(output) if output.exists() else []
        by_id = {row["id"]: row for row in rows}
        for result in results:
            if result["id"] not in by_id:
                raise ValueError("Resume contains unrelated IDs")
            _validate_result(result, by_id[result["id"]], keys[result["id"]])
        done = {result["id"] for result in results}
        with output.open("a", encoding="utf-8", buffering=1) as stream, ThreadPoolExecutor(max_workers=min(4, max(1, workers))) as pool:
            futures = [pool.submit(_evaluate, row, docs, reference[row["id"]], candidate[row["id"]], config, secrets, keys[row["id"]])
                       for row in rows if row["id"] not in done]
            for future in as_completed(futures):
                result = future.result()
                _validate_result(result, by_id[result["id"]], keys[result["id"]])
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
                print(json.dumps({key: result.get(key) for key in ("id", "status", "score")}), flush=True)
        results.sort(key=lambda result: meta["ids"].index(result["id"]))
        # Pilot scope prevents the legacy selector from declaring a candidate
        # based on beating a common reference. Cross-candidate inference is separate.
        summary = selection_summary(results, scope="pilot")
        summary.update(protocol=PROTOCOL, scope="common_reference", supported_direction=None,
                       winner=None, candidate_selection_supported=False, not_psb_prediction=True,
                       reference_role=REFERENCE_ROLE, reference_package_sha256=reference_meta["package_sha256"],
                       candidate_package_sha256=candidate_meta["package_sha256"], ids=meta["ids"],
                       status_counts=dict(sorted(Counter(row["status"] for row in results).items())),
                       technical_error_count=sum(row["status"] == "judge_error" for row in results),
                       unresolved_ids=[row["id"] for row in results if row["status"] != "ok"],
                       by_question_family=summary["by_family_id"], details_sha256=sha(output), resume_key=run_key,
                       interpretation="Common fixed-reference score only; a score above 0.5 does not establish a winning candidate.")
        write_json(output.with_suffix(".summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "kb", "reference-run", "candidate-run", "services", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        summary = run(**vars(args))
    except Exception as exc:
        # No arbitrary exception text: network/config errors may contain secrets.
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
