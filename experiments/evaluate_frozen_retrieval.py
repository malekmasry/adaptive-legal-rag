"""Evaluate a frozen question/gold JSON against the current offline retriever.

This is an evaluation-only script. It never changes the product's retrieval rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adaptive_pipeline import AdaptivePipeline  # noqa: E402
from legal_core import LegalAssistant, ModelProxy, focused_search_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/kb.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing evaluation")

    cases = json.loads(args.questions.read_text(encoding="utf-8"))["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Empty or duplicate case IDs")
    engine = LegalAssistant(args.archive, ModelProxy(), use_embeddings=False)
    available = {source.id for source in engine.sources}
    for case in cases:
        if not case["gold_groups"] or any(not set(group) & available for group in case["gold_groups"]):
            raise ValueError(f"Gold source missing for {case['id']}")

    report = {"questions": str(args.questions), "archive": str(args.archive),
              "method": "lexical + explicit links + neighbor expansion + context packing",
              "question_count": len(cases), "document_count": len(engine.documents),
              "source_count": len(engine.sources), "link_count": len(engine.links), "cases": []}
    for case in cases:
        started = perf_counter()
        raw = [engine.sources[index] for index in engine._lexical(focused_search_question(case["question"]))]
        retrieved = engine.retrieve(case["question"])
        pipeline = AdaptivePipeline(engine)
        pipeline.pack(pipeline.enrich(retrieved))
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        raw_ids = [source.id for source in raw]
        retrieved_ids = [source.id for source in retrieved]
        context_ids = list(pipeline.evidence)
        ranks = [min((raw_ids.index(source_id) + 1 for source_id in group if source_id in raw_ids), default=None)
                 for group in case["gold_groups"]]
        coverage = lambda ids: all(set(group) & set(ids) for group in case["gold_groups"])
        report["cases"].append({"id": case["id"], "question": case["question"],
                                "gold_groups": case["gold_groups"], "raw_gold_ranks": ranks,
                                "raw_all_at_16": all(rank is not None and rank <= 16 for rank in ranks),
                                "retrieved_all": coverage(retrieved_ids),
                                "context_all": coverage(context_ids),
                                "retrieved_ids": retrieved_ids,
                                "context_ids": context_ids, "elapsed_ms": elapsed_ms})
        print(f"{case['id']} context={'OK' if report['cases'][-1]['context_all'] else 'MISS'} "
              f"raw_ranks={ranks} time={elapsed_ms:.0f}ms", flush=True)
    report["summary"] = {key: sum(bool(item[key]) for item in report["cases"])
                         for key in ("raw_all_at_16", "retrieved_all", "context_all")}
    report["summary"]["mean_elapsed_ms"] = round(sum(item["elapsed_ms"] for item in report["cases"]) / len(cases), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
