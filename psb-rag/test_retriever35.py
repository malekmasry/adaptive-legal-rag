"""Adapter integration/security tests; no external API calls."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from corpus import ingest
from model_api import API, ModelError
from reasoning import evidence_payload, validate_claim
from retriever35 import Retriever, Proxy
from retriever35_core import CURRENT
from schemas import Plan, Claim, Citation
from settings import Settings
from source_audit import load_audit

ROOT = Path(__file__).resolve().parent


class TestRetriever35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.settings = Settings(data_dir=ROOT/'data', cache_dir=Path(cls.tmp.name), use_rerank=False)
        cls.docs, cls.audit_chunks, cls.relations = ingest(cls.settings.data_dir)
        cls.audit = load_audit(cls.audit_chunks, ROOT)
        cls.r = Retriever(cls.docs, cls.audit_chunks, cls.relations, None, cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_raw_offsets_and_342_genuine_cached_document_vectors(self):
        self.assertEqual(len(self.r.chunks),342)
        self.assertEqual(self.r.vectors.shape,(342,1024))
        self.assertEqual(len(self.r.text_hashes),342)
        docs = {d.id:d for d in self.docs}
        for chunk in self.r.chunks:
            self.assertEqual(chunk.text,docs[chunk.doc_id].raw[chunk.start:chunk.end])

    def test_v8_instruction_audit_survives_new_boundaries(self):
        self.assertIsNotNone(self.audit)
        all_by_id = {c.id:c for c in self.r.chunks}
        span_chunk = next(c for c in self.audit_chunks if c.instruction_spans)
        span = span_chunk.instruction_spans[0]
        target = next(c for c in self.r.chunks if c.doc_id == span_chunk.doc_id and
                      c.start <= span['start'] and c.end >= span['end'])
        # Force a retrieved contaminated chunk; the adapter must keep the raw
        # source and carry the audit into v8's claim validator.
        with patch.object(self.r.engine,'retrieve',return_value=[self.r.engine.by_id[target.id]]):
            selected,_ = self.r.search(Plan(question='проверка'))
        chunk = next(c for c in selected if c.id==target.id)
        self.assertEqual(chunk.text,all_by_id[target.id].text)
        self.assertIn(span['text'],chunk.text)
        items,_ = evidence_payload([chunk],self.r.docs,18000)
        self.assertIn(span['text'],items[0]['instruction_texts'])
        claim = Claim(text=span['text'],citations=[Citation(chunk_id=chunk.id,quote=span['text'])])
        issues = validate_claim(claim,{chunk.id:items[0]})
        self.assertIn('quote belongs to instruction span, not a legal rule', issues)

    def test_no_v8_date_or_entity_filter_after_donor_ranking(self):
        question='Какая арендная плата по А-08/2024?'
        a,_ = self.r.search(Plan(question=question))
        b,_ = self.r.search(Plan(question=question,entity='wrong-entity',as_of='1900-01-01',date_role='event'))
        self.assertEqual([c.id for c in a],[c.id for c in b])

    def test_concurrent_traces_and_chunks_are_isolated(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:self.r.search(Plan(question='Арендная плата А-08/2024?')),range(8)))
        self.assertIsNone(CURRENT.get())
        first=results[0][0][0]
        self.assertTrue(all([c.id for c in rows]==results[0][1]['selected'] for rows,_ in results))
        self.assertIsNot(first,results[1][0][0])

    def test_changed_model_does_not_reuse_bundled_vectors(self):
        s=replace(self.settings,emb_model='another-model')
        r=Retriever(self.docs,self.audit_chunks,self.relations,None,s)
        self.assertIsNone(r.vectors)
        self.assertNotEqual(r.cache_path.name,self.r.cache_path.name)

    def test_changed_corpus_does_not_reuse_bundled_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'new.txt').write_text('Статья 1. Совсем другой корпус.',encoding='utf8')
            s=replace(self.settings,data_dir=root)
            d,c,r=ingest(root);adapter=Retriever(d,c,r,None,s)
            self.assertIsNone(adapter.vectors)
            self.assertEqual(len(adapter.chunks),1)

    def test_corrupt_cache_is_rejected(self):
        s=replace(self.settings,emb_model='corrupt-test')
        r=Retriever(self.docs,self.audit_chunks,self.relations,None,s)
        np.savez_compressed(r.cache_path,vectors=np.full((342,2),np.nan),text_hashes=np.array(r.text_hashes))
        other=Retriever(self.docs,self.audit_chunks,self.relations,None,s)
        self.assertIsNone(other.vectors)

    def test_reranker_transport_preserves_full_pool_and_raw_query(self):
        seen=[]
        def handler(request):
            payload=json.loads(request.content);seen.append(payload)
            return httpx.Response(200,json={'results':[{'index':2,'relevance_score':.9},
                {'index':0,'relevance_score':.3},{'index':1,'relevance_score':.1}]})
        s=replace(self.settings,base_url='https://offline.invalid',api_key='offline',use_rerank=True)
        api=API(s,transport=httpx.MockTransport(handler))
        try:
            result=Proxy(api,s).rerank_with_scores('Вопрос?', ['a','b','c'])
            self.assertEqual(result,[(2,.9),(0,.3),(1,.1)])
            self.assertEqual(seen[0]['query'],'Вопрос?')
            self.assertEqual(seen[0]['top_n'],3)
            self.assertEqual(seen[0]['model'],'Qwen/Qwen3-Reranker-8B')
        finally:api.close()

    def test_separate_endpoint_never_receives_generator_key(self):
        s=replace(self.settings,rerank_url='https://offline.invalid',rerank_key='',use_rerank=True)
        with self.assertRaises(ModelError):Proxy(None,s).rerank_with_scores('q',['a'])
        s=replace(s,emb_url='https://offline.invalid',emb_key='')
        with self.assertRaises(ModelError):Proxy(None,s).embeddings(['a'])


if __name__=='__main__':unittest.main()
