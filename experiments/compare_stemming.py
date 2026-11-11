"""Offline paired retrieval comparison; no embeddings, reranker, graph or LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legal_core import LegalAssistant, ModelProxy, focused_search_question, tokens
from ru_stemmer import stem_russian


class OfflineProxy(ModelProxy):
    def __init__(self):
        self.base = self.key = self.llm = self.rerank_model = ""

    def post(self, *args, **kwargs):
        raise AssertionError("Network calls are forbidden in this experiment")


class PlainBM25:
    def __init__(self, texts, stemmed=False):
        self.stemmed = stemmed
        self.terms = [Counter(self.tokenize(text)) for text in texts]
        self.df = Counter(term for document in self.terms for term in document)
        self.lengths = [sum(document.values()) for document in self.terms]
        self.average = sum(self.lengths) / max(1, len(texts))

    def tokenize(self, text):
        words = tokens(text)
        return [stem_russian(word) for word in words] if self.stemmed else words

    def rank(self, query):
        terms = set(self.tokenize(query))
        scores = []
        count = len(self.terms)
        for document, length in zip(self.terms, self.lengths):
            score = 0.0
            for term in sorted(terms):
                frequency = document[term]
                if frequency:
                    idf = math.log(1 + (count - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.average))
            scores.append(score)
        return sorted(range(count), key=lambda index: (-scores[index], index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an experiment")
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != payload["kb_sha256"]:
        raise ValueError("Corpus differs from gold manifest")
    engines, initialization = {}, {}
    for mode in ("legacy", "snowball"):
        stem_russian.cache_clear()
        started = perf_counter()
        engines[mode] = LegalAssistant(args.archive, OfflineProxy(), use_embeddings=False, lexical_stemmer=mode)
        initialization["current_" + mode] = (perf_counter() - started) * 1000
    engine = engines["legacy"]
    manifest = [{"id": s.id, "path": s.path, "text": s.text} for s in engine.sources]
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if digest != payload["source_manifest_sha256"]:
        raise ValueError("Source IDs/text differ from gold manifest")
    assert engine.search_texts == engines["snowball"].search_texts
    cases = payload["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        for group in case["gold_groups"]:
            assert group and all(sid in engine.by_id for sid in group)
    methods = {"current_legacy": engines["legacy"]._lexical, "current_snowball": engines["snowball"]._lexical}
    for mode in (False, True):
        stem_russian.cache_clear()
        name = "plain_snowball" if mode else "plain_bm25"
        started = perf_counter()
        methods[name] = PlainBM25(engine.search_texts, mode).rank
        initialization[name] = (perf_counter() - started) * 1000
    rows = {name: [] for name in methods}
    for case in cases:
        query = focused_search_question(case["question"])
        for name, rank in methods.items():
            started = perf_counter()
            ids = [engine.sources[i].id for i in rank(query)]
            elapsed = (perf_counter() - started) * 1000
            positions = {sid: i + 1 for i, sid in enumerate(ids)}
            ranks = [min(positions[sid] for sid in group) for group in case["gold_groups"]]
            row = {"id": case["id"], "dataset": case["dataset"], "split": case["split"],
                   "ranks": ranks, "latency_ms": elapsed, "top64": ids[:64]}
            if ranks:
                for k in (5, 12, 16, 64):
                    row[f"all@{k}"] = max(ranks) <= k
                    row[f"recall@{k}"] = sum(r <= k for r in ranks) / len(ranks)
                row["mrr"] = 1 / min(ranks)
            rows[name].append(row)
    summaries = {}
    for name, results in rows.items():
        summaries[name] = {}
        for subset in ("all", "original54", "fresh_frozen", "additional25"):
            selected = [r for r in results if r["ranks"] and (subset == "all" or subset in (r["dataset"], r["split"]))]
            summaries[name][subset] = {"n": len(selected), **{f"all@{k}": sum(r[f"all@{k}"] for r in selected) for k in (5, 12, 16, 64)},
                "recall@64": statistics.mean(r["recall@64"] for r in selected),
                "mrr": statistics.mean(r["mrr"] for r in selected),
                "mean_ms": statistics.mean(r["latency_ms"] for r in selected),
                "median_ms": statistics.median(r["latency_ms"] for r in selected)}
    changes = {}
    for baseline, variant in (("plain_bm25", "plain_snowball"), ("current_legacy", "current_snowball")):
        changes[variant] = {}
        for k in (12, 64):
            gain, loss = [], []
            for a, b in zip(rows[baseline], rows[variant]):
                if not a["ranks"]:
                    continue
                if not a[f"all@{k}"] and b[f"all@{k}"]:
                    gain.append(a["id"])
                if a[f"all@{k}"] and not b[f"all@{k}"]:
                    loss.append(a["id"])
            changes[variant][f"all@{k}"] = {"improved": gain, "regressed": loss}
    report = {"method": "Raw retrieval only; no API, graph expansion, reranking or generation",
              "plain_inputs": "search_texts: document, contract number, parties, parent scope, source text",
              "current_inputs": "Unchanged production scoring; only search_terms morphology differs",
              "k1": 1.2, "b": 0.75, "source_count": len(engine.sources), "question_count": len(cases),
              "gold_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(), "source_manifest_sha256": digest,
              "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("legal_core.py", "ru_stemmer.py")},
              "initialization_ms": initialization, "summaries": summaries, "changes": changes, "results": rows,
              "excluded_no_gold": [c["id"] for c in cases if not c["gold_groups"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "changes": changes, "excluded": report["excluded_no_gold"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
