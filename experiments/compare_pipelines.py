"""Save reviewable API outputs for direct/adaptive/oracle comparisons.

No automatic truth score: status=answered is not correctness. Corpus, model,
settings and per-request metrics are recorded so a reviewer can compare runs.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_pipeline import AdaptivePipeline
from legal_core import LegalAssistant, ModelProxy, setting
from single_pass import SinglePassPipeline, pack_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("single_pass", "adaptive", "oracle"), default=["single_pass"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ask-key", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing report")
    if args.ask_key:
        os.environ["LLM_API_KEY"] = getpass.getpass("API key (not saved): ")
    cases = json.loads(args.questions.read_text(encoding="utf-8-sig"))["cases"]
    if args.limit:
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        cases = cases[:args.limit]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Empty or duplicate cases")
    proxy = ModelProxy()
    if not proxy.ready:
        raise SystemExit("Configure LLM_BASE_URL, LLM_API_KEY and LLM_MODEL")
    engine = LegalAssistant(args.archive, proxy)
    oracle = {}
    for case in cases:
        if "oracle" not in args.modes:
            continue
        groups = case.get("gold_groups") or [[sid] for sid in case.get("gold_source_ids", [])]
        if not groups or any(not any(sid in engine.by_id for sid in group) for group in groups):
            raise ValueError(f"Missing valid gold sources for {case['id']}; recheck IDs after changing chunking")
        selected = list(dict.fromkeys(next(sid for sid in group if sid in engine.by_id) for group in groups))
        oracle[case["id"]] = [engine.by_id[sid] for sid in selected]
        if {record["id"] for record in pack_context(engine, oracle[case["id"]])} != set(selected):
            raise ValueError(f"Oracle sources exceed context limits for {case['id']}")
    settings = {key: setting(key) for key in ("TOP_K", "RETRIEVAL_CANDIDATES", "CONTEXT_MAX_CHARS",
                "CONTEXT_MAX_SOURCES", "PIPELINE_TIMEOUT_SECONDS", "MAX_TOKENS", "STRUCTURED_MAX_TOKENS",
                "USE_EMBEDDINGS", "EMB_MODEL", "RERANKER_MODEL", "TEMPERATURE")}
    report = {"model": proxy.llm, "settings_overrides": settings,
              "implementation_sha256": hashlib.sha256(b"".join(
                  (Path(__file__).resolve().parents[1] / name).read_bytes()
                  for name in ("legal_core.py", "single_pass.py", "adaptive_pipeline.py", "rag_safety.py", "hallugraph.py"))).hexdigest(),
              "corpus_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              "documents": len(engine.documents), "sources": len(engine.sources), "runs": [],
              "note": "Status is operational; correctness must be evaluated independently. Oracle uses annotated sources."}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["summary"] = {mode: {"count": len(rows := [row for row in report["runs"] if row["mode"] == mode]),
                                    "statuses": dict(Counter(row["status"] for row in rows)),
                                    "prompt_tokens": sum(row["diagnostics"].get("prompt_tokens", 0) for row in rows),
                                    "completion_tokens": sum(row["diagnostics"].get("completion_tokens", 0) for row in rows)}
                             for mode in dict.fromkeys(args.modes)}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)

    for case in cases:
        for mode in dict.fromkeys(args.modes):
            if mode == "adaptive":
                answer = AdaptivePipeline(engine).run(case["question"])
            else:
                answer = SinglePassPipeline(engine).run(case["question"], sources_override=oracle.get(case["id"]) if mode == "oracle" else None)
            report["runs"].append({"id": case["id"], "question": case["question"], "mode": mode,
                                   **answer.public(), "manual_correctness": None})
            save()
            print(f"{case['id']} {mode}: {answer.status}, calls={answer.diagnostics.get('pipeline_calls')}", flush=True)


if __name__ == "__main__":
    main()
