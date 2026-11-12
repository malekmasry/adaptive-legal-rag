"""Multi-query isolation, identifier preservation, batching, fusion and deadlines."""
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch
import zipfile

from legal_core import LegalAssistant, ModelProxy
from multi_query import dense_candidates, reformulations, validated_queries
from rag_telemetry import CURRENT, DEADLINE, Trace, remaining_seconds


def completion(queries):
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"queries": queries}, ensure_ascii=False)}}]}


class MultiQueryTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"DENSE_MULTI_QUERY": "1", "RETRIEVAL_MODE": "conditional",
                                     "USE_EMBEDDINGS": "0", "RETRIEVAL_CANDIDATES": "16"})
        env.start()
        self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.archive = Path(temp.name) / "kb.zip"
        with zipfile.ZipFile(self.archive, "w") as bundle:
            for i in range(20):
                bundle.writestr(f"rule{i}.txt", f"Правило {i}\n1.1. Срок оплаты товара {i}.")
        self.proxy = ModelProxy()
        self.proxy.base, self.proxy.key, self.proxy.llm, self.proxy.rerank_model = "http://stub", "stub", "llm", "rerank"
        self.engine = LegalAssistant(self.archive, self.proxy, use_embeddings=False)
        self.engine.vectors = [[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(len(self.engine.sources))]
        self.trace = Trace(deadline_seconds=90)
        token = CURRENT.set(self.trace)
        self.addCleanup(CURRENT.reset, token)

    def test_strong_route_does_not_rewrite_or_embed(self):
        with patch.object(self.proxy, "rerank_with_scores", return_value=[(i, 0.9) for i in range(16)]), \
                patch.object(self.proxy, "post") as post, patch.object(self.proxy, "embeddings") as embed:
            self.engine.retrieve("Когда платить?")
        post.assert_not_called()
        embed.assert_not_called()

    def test_identifiers_numbers_duplicates_and_original_preserved(self):
        q = 'Когда «Альфа-Трейд» получает оплату по П-17/2024 после 01.10.2024?'
        good = 'Каков срок платежа «Альфа-Трейд» по П-17/2024 после 01.10.2024?'
        bad = 'Оплата «Альфа-Трейдинг» по П-17/2024 после 01.10.2024 через 30 дней?'
        self.assertEqual(validated_queries(q, {"queries": [good, bad]}), [q, good])
        self.assertEqual(validated_queries(q, {"queries": [q, q]}), [q])
        self.assertEqual(validated_queries(q, {"queries": [good, good]}), [q, good])

    def test_single_bounded_rewrite_receives_conversation_and_restores_stage(self):
        self.trace.stage = "dense"
        outer = DEADLINE.set(perf_counter() + 20)
        original_deadline = DEADLINE.get()
        try:
            def respond(endpoint, payload, **kwargs):
                self.assertEqual(endpoint, "/chat/completions")
                self.assertEqual(kwargs["attempts"], 1)
                self.assertLessEqual(remaining_seconds(), 6)
                self.assertEqual(self.trace.stage, "query_rewrite")
                self.assertEqual(json.loads(payload["messages"][1]["content"])["conversation"], "История")
                return completion(["Когда платить за товар?", "Каков срок расчетов?"])
            with patch.object(self.proxy, "post", side_effect=respond) as post:
                result = reformulations(self.proxy, "Срок оплаты?", "История")
            self.assertEqual(len(result), 3)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(DEADLINE.get(), original_deadline)
            self.assertEqual(self.trace.stage, "dense")
        finally:
            DEADLINE.reset(outer)

    def test_invalid_or_failed_rewrite_uses_original_dense(self):
        for response in (RuntimeError("timeout"), {}, completion(["одна"]),
                         {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}):
            with self.subTest(response=response), patch.object(self.proxy, "post", side_effect=[response]), \
                    patch.object(self.engine, "_dense", return_value=[3, 2, 1]) as dense:
                self.assertEqual(dense_candidates(self.engine, "Когда платить?", "", 2), [3, 2])
            dense.assert_called_once_with("Когда платить?")

    def test_little_remaining_time_skips_rewrite(self):
        deadline = DEADLINE.set(perf_counter() + 1)
        try:
            with patch.object(self.proxy, "post") as post:
                self.assertEqual(reformulations(self.proxy, "Вопрос?", ""), ["Вопрос?"])
            post.assert_not_called()
        finally:
            DEADLINE.reset(deadline)

    def test_batch_embeddings_and_dedup_identical_results(self):
        original = [i for i in range(len(self.engine.sources)) if i % 2] + [i for i in range(len(self.engine.sources)) if not i % 2]
        with patch.object(self.proxy, "post", return_value=completion(["Когда платить за товар?", "Каков срок расчетов?"])), \
                patch.object(self.proxy, "embeddings", return_value=[[1.0, 0.0]] * 3) as embed:
            chosen = dense_candidates(self.engine, "Срок оплаты?", "", 8)
        self.assertEqual(chosen, original[:8])
        self.assertEqual(embed.call_count, 1)
        self.assertEqual(len(embed.call_args.args[0]), 3)
        self.assertEqual(embed.call_args.args[0][0], "Срок оплаты?")
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_new_query_adds_candidates_and_reranks_original_question_once(self):
        pools, questions = [], []
        def rank(query, texts):
            pools.append(texts)
            questions.append(query)
            return [(i, 0.1 if len(pools) == 1 else 0.9) for i in range(len(texts))]
        with patch.object(self.proxy, "post", return_value=completion(["Когда платить за товар?", "Каков срок расчетов?"])), \
                patch.object(self.proxy, "embeddings", return_value=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), \
                patch.object(self.proxy, "rerank_with_scores", side_effect=rank):
            found = self.engine.retrieve("Срок оплаты?")
        self.assertEqual(questions, ["Срок оплаты?", "Срок оплаты?"])
        self.assertEqual(len(pools), 2)
        self.assertEqual(len(pools[1]), 16)
        self.assertTrue(found)
        event = next(e for e in self.trace.events if e["stage"] == "multi_query_dense")
        self.assertGreater(event["unique_candidates"], 16)
        self.assertEqual(event["selected_candidates"], 16)

    def test_batch_embedding_failure_falls_back_to_single_query(self):
        with patch.object(self.proxy, "post", return_value=completion(["Когда платить за товар?", "Каков срок расчетов?"])), \
                patch.object(self.proxy, "embeddings", side_effect=RuntimeError("unavailable")), \
                patch.object(self.engine, "_dense", return_value=[1, 2, 3]) as dense:
            self.assertEqual(dense_candidates(self.engine, "Срок оплаты?", "", 2), [1, 2])
        dense.assert_called_once_with("Срок оплаты?")

    def test_dense_reranker_failure_retains_reranked_bm25(self):
        order = self.engine._lexical("Срок оплаты?")[:16]
        with patch.object(self.proxy, "post", return_value=completion(["Когда платить за товар?", "Каков срок расчетов?"])), \
                patch.object(self.proxy, "embeddings", return_value=[[1.0, 0.0]] * 3), \
                patch.object(self.proxy, "rerank_with_scores", side_effect=[
                    [(i, 0.1) for i in reversed(range(16))], RuntimeError("unavailable")]):
            found = self.engine.retrieve("Срок оплаты?", limit=4)
        self.assertEqual([s.id for s in found], [self.engine.sources[i].id for i in order[::-1][:4]])


if __name__ == "__main__":
    unittest.main()
