"""Route selection, independent candidate pools and preservation on API failure."""
import os
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from legal_core import LegalAssistant, ModelProxy
from rag_telemetry import CURRENT, DEADLINE, Trace


class ConditionalRetrievalTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RETRIEVAL_MODE": "conditional", "CONDITIONAL_DENSE_THRESHOLD": "0.5",
                                     "RETRIEVAL_CANDIDATES": "16", "USE_EMBEDDINGS": "0"})
        env.start()
        self.addCleanup(env.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.archive = Path(temp.name) / "kb.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            for i in range(40):
                archive.writestr(f"rule_{i}.txt", f"Rule {i}\n1.1. Unique provision number {i}.")
        self.proxy = ModelProxy()
        self.proxy.base, self.proxy.key, self.proxy.rerank_model = "http://stub", "stub", "reranker"
        self.proxy.embed_base = self.proxy.rerank_base = ""
        self.engine = LegalAssistant(self.archive, self.proxy, use_embeddings=False)
        self.engine.vectors = [[1.0, 0.0] for _ in self.engine.sources]
        self.lexical = list(range(len(self.engine.sources)))
        self.dense = list(reversed(self.lexical))
        self.trace = Trace()
        token = CURRENT.set(self.trace)
        self.addCleanup(CURRENT.reset, token)

    @staticmethod
    def scored(count, score, field="relevance_score"):
        return {"results": [{"index": i, field: score} for i in reversed(range(count))]}

    def route(self):
        return next(event for event in reversed(self.trace.events) if event["stage"] == "retrieval_route")

    def retrieve(self):
        with patch.object(self.engine, "_lexical", return_value=self.lexical):
            return self.engine.retrieve("Question", limit=4)

    def test_confident_lexical_never_embeds_query_or_calls_dense(self):
        with patch.object(self.proxy, "post", return_value=self.scored(16, 0.8)) as post, \
                patch.object(self.engine, "_dense") as dense, patch.object(self.proxy, "embeddings") as embed:
            found = self.retrieve()
        self.assertEqual([source.id for source in found], [self.engine.sources[i].id for i in (15, 14, 13, 12)])
        dense.assert_not_called()
        embed.assert_not_called()
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.route()["status"], "lexical_confident")

    def test_low_score_uses_separate_dense_pool_without_lexical_fusion(self):
        with patch.object(self.proxy, "post", side_effect=[self.scored(16, 0.1), self.scored(16, 0.9)]) as post, \
                patch.object(self.engine, "_dense", return_value=self.dense) as dense:
            found = self.retrieve()
        dense.assert_called_once_with("Question")
        first, second = [call.args[1]["documents"] for call in post.call_args_list]
        self.assertEqual(first, [self.engine.search_texts[i] for i in self.lexical[:16]])
        self.assertEqual(second, [self.engine.search_texts[i] for i in self.dense[:16]])
        self.assertTrue(set(first).isdisjoint(second))
        self.assertEqual([source.id for source in found], [self.engine.sources[i].id for i in self.dense[:16][::-1][:4]])
        self.assertEqual(self.route()["route"], "dense")
        self.assertIn("rerank_lexical", self.trace.timings)
        self.assertIn("rerank_dense", self.trace.timings)

    def test_maximum_score_and_exact_threshold_keep_lexical(self):
        response = self.scored(16, 0.1)
        response["results"][-1]["relevance_score"] = 0.5
        with patch.object(self.proxy, "post", return_value=response), patch.object(self.engine, "_dense") as dense:
            self.retrieve()
        dense.assert_not_called()
        self.assertEqual(self.route()["rerank_score"], 0.5)

    def test_unusable_scores_keep_reranked_lexical_evidence(self):
        for value in (None, "0.1", float("nan"), float("inf"), -4.0, 2.0, True):
            with self.subTest(value=value), patch.object(self.proxy, "post", return_value=self.scored(16, value)), \
                    patch.object(self.engine, "_dense") as dense:
                found = self.retrieve()
                dense.assert_not_called()
                self.assertEqual(found[0].id, self.engine.sources[15].id)
                self.assertEqual(self.route()["status"], "scores_unavailable")

    def test_partial_rerank_response_does_not_trigger_dense(self):
        with patch.object(self.proxy, "post", return_value=self.scored(1, 0.1)), \
                patch.object(self.engine, "_dense") as dense:
            self.retrieve()
        dense.assert_not_called()
        self.assertEqual(self.route()["status"], "scores_unavailable")

    def test_dense_failures_preserve_already_reranked_lexical(self):
        for failure in ("embedding", "reranker", "empty"):
            responses = [self.scored(16, 0.1), RuntimeError("unavailable")]
            with self.subTest(failure=failure), patch.object(self.proxy, "post", side_effect=responses), \
                    patch.object(self.engine, "_dense", return_value=[] if failure == "empty" else self.dense,
                                 side_effect=RuntimeError("unavailable") if failure == "embedding" else None):
                found = self.retrieve()
                self.assertEqual([source.id for source in found], [self.engine.sources[i].id for i in (15, 14, 13, 12)])
                self.assertEqual(self.route()["route"], "lexical")
                self.assertNotEqual(self.route()["status"], "low_lexical_score")

    def test_lexical_rerank_failure_keeps_original_lexical(self):
        with patch.object(self.proxy, "post", side_effect=RuntimeError("unavailable")), \
                patch.object(self.engine, "_dense") as dense:
            found = self.retrieve()
        dense.assert_not_called()
        self.assertEqual(found[0].id, self.engine.sources[0].id)
        self.assertEqual(self.route()["status"], "lexical_rerank_failed")

    def test_missing_index_does_not_embed_query(self):
        self.engine.vectors = None
        with patch.object(self.proxy, "post", return_value=self.scored(16, 0.1)), \
                patch.object(self.proxy, "embeddings") as embed:
            self.retrieve()
        embed.assert_not_called()
        self.assertEqual(self.route()["status"], "dense_index_unavailable")

    def test_no_reranker_skips_index_and_query_embeddings(self):
        self.proxy.rerank_model = ""
        with patch.object(self.proxy, "embeddings") as embed, patch.object(self.proxy, "post") as post:
            engine = LegalAssistant(self.archive, self.proxy, use_embeddings=True)
            self.assertTrue(engine.retrieve("Question"))
        embed.assert_not_called()
        post.assert_not_called()
        self.assertIsNone(engine.vectors)
        self.assertEqual(self.route()["status"], "reranker_unavailable")

    def test_index_is_built_once_and_queries_only_embed_on_weak_route(self):
        def respond(endpoint, payload, **kwargs):
            if endpoint == "/embeddings":
                return {"data": [{"index": i, "embedding": [1, 0]} for i in range(len(payload["input"]))]}
            return self.scored(len(payload["documents"]), 0.1 if payload["query"] == "weak" else 0.9)
        with patch.object(self.proxy, "post", side_effect=respond) as post:
            engine = LegalAssistant(self.archive, self.proxy, use_embeddings=True)
            index_calls = sum(call.args[0] == "/embeddings" for call in post.call_args_list)
            engine.retrieve("strong")
            engine.retrieve("weak")
            engine.retrieve("strong")
        embedding_calls = [call for call in post.call_args_list if call.args[0] == "/embeddings"]
        self.assertEqual(len(embedding_calls), index_calls + 1)
        self.assertEqual(embedding_calls[-1].args[1]["input"], ["weak"])
        self.assertEqual(sum(call.args[0] == "/rerank" for call in post.call_args_list), 4)

    def test_expired_dense_deadline_preserves_lexical(self):
        with patch.object(self.proxy, "rerank_with_scores", return_value=[(i, 0.1) for i in reversed(range(16))]), \
                patch("legal_core.httpx.AsyncClient") as client:
            deadline = DEADLINE.set(0)
            try:
                found = self.retrieve()
            finally:
                DEADLINE.reset(deadline)
        client.assert_not_called()
        self.assertEqual(found[0].id, self.engine.sources[15].id)
        self.assertEqual(self.route()["status"], "dense_search_failed")

    def test_concurrent_questions_keep_independent_scores(self):
        def respond(endpoint, payload, **kwargs):
            return self.scored(len(payload["documents"]), 0.1 if payload["query"] == "weak" else 0.9)

        def work(query):
            trace = Trace()
            token = CURRENT.set(trace)
            try:
                self.engine.retrieve(query)
                return next(event for event in trace.events if event["stage"] == "retrieval_route")
            finally:
                CURRENT.reset(token)

        with patch.object(self.proxy, "post", side_effect=respond), patch.object(self.engine, "_dense", return_value=self.dense), \
                ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(work, ["strong", "weak", "strong", "weak"]))
        self.assertEqual([item["route"] for item in results], ["lexical", "dense", "lexical", "dense"])

    def test_score_aliases_keep_legacy_index_interface(self):
        for container in ("results", "data"):
            for field in ("score", "relevance_score"):
                response = {container: [{"index": 1, field: 0.8}, {"index": 0, field: 0.1}]}
                with self.subTest(container=container, field=field), patch.object(self.proxy, "post", return_value=response):
                    self.assertEqual(self.proxy.rerank("Q", ["a", "b"]), [1, 0])
                    self.assertEqual(self.proxy.rerank_with_scores("Q", ["a", "b"]), [(1, 0.8), (0, 0.1)])


if __name__ == "__main__":
    unittest.main()
