import os,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
from focused_context import ScopeRetriever
from model_api import ModelError
from pipeline import Pipeline
from schemas import Plan
from settings import Settings


class ConditionalTests(unittest.TestCase):
    def fixture(self,score=.8,failure=None):
        r=object.__new__(ScopeRetriever)
        r.s=SimpleNamespace(top_k=40,rerank_n=2,use_rerank=True,retrieval_mode='conditional')
        r.chunks=[SimpleNamespace(id=x,doc_id=x,search_text=x,parent_id=None,ordinal=0) for x in ('A','B','C','D')]
        r.all={c.id:c for c in r.chunks}
        r.docs={x:SimpleNamespace(filename=x+'.txt',metadata={'entity':''}) for x in r.all}
        r.lexical=lambda q,pool,limit=None:[('A',3),('B',2)][:limit or 40]
        events=[]
        def dense(queries,pool):
            events.append(('embed',tuple(queries)))
            if failure=='embedding':raise ModelError('embedding failed')
            return [[('C',.8),('D',.7)] for _ in queries]
        def rerank(query,texts,limit):
            events.append(('rerank',tuple(texts),limit))
            if failure=='lexical' or (failure=='dense_rerank' and texts[0]=='C'):
                raise ModelError('rerank failed')
            return [(0,score if texts[0]=='A' else .9),(1,.1)]
        r.dense_many=dense;r.api=SimpleNamespace(rerank=rerank,deadline=time.monotonic()+1000)
        return r,events

    def test_confident_lexical_never_embeds_query(self):
        r,events=self.fixture(.5)
        chunks,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual([c.id for c in chunks],['A','B'])
        self.assertEqual(events,[('rerank',('A','B'),2)])
        self.assertEqual(trace['route_reason'],'lexical_confident')

    def test_weak_lexical_is_replaced_not_fused(self):
        r,events=self.fixture(.49)
        chunks,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual([c.id for c in chunks],['C','D'])
        self.assertEqual([e[0] for e in events],['rerank','embed','rerank'])
        self.assertEqual(trace['route'],'dense')

    def test_each_alternate_failure_retains_reranked_lexical(self):
        for failure in ('embedding','dense_rerank'):
            r,_=self.fixture(.2,failure)
            chunks,trace=r.search(Plan(question='обычный вопрос'))
            self.assertEqual([c.id for c in chunks],['A','B'])
            self.assertEqual(trace['route_reason'],'alternate_failed_kept_lexical')
            self.assertTrue(trace['degraded'])

    def test_failed_or_disabled_rerank_does_not_guess_confidence(self):
        r,events=self.fixture(.2,'lexical')
        _,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual(trace['route_reason'],'lexical_rerank_failed')
        self.assertEqual(len(events),1)
        r,events=self.fixture();r.s.use_rerank=False
        chunks,_=r.search(Plan(question='обычный вопрос'))
        self.assertEqual(events,[]);self.assertEqual([c.id for c in chunks],['A','B'])

    def test_non_probability_scores_keep_lexical(self):
        r,events=self.fixture(7)
        _,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual(trace['route_reason'],'score_scale_unavailable');self.assertEqual(len(events),1)

    def test_deadline_preserves_first_route(self):
        r,events=self.fixture(.2);r.api.deadline=time.monotonic()+10
        chunks,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual(trace['route_reason'],'alternate_skipped_deadline')
        self.assertEqual([c.id for c in chunks],['A','B']);self.assertEqual(len(events),1)

    def test_empty_lexical_can_recover_dense(self):
        r,events=self.fixture();r.lexical=lambda *a,**k:[]
        chunks,trace=r.search(Plan(question='обычный вопрос'))
        self.assertEqual([c.id for c in chunks],['C','D'])
        self.assertEqual(trace['route_reason'],'empty_lexical')
        self.assertEqual([e[0] for e in events],['embed','rerank'])

    def test_subquestion_evidence_is_still_retained(self):
        r,_=self.fixture(.2)
        chunks,_=r.search(Plan(question='основной вопрос',subquestions=['другой вопрос']))
        self.assertEqual([c.id for c in chunks],['C','D','A','B'])

    def test_default_mode_and_lazy_index_keep_existing_override(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(Settings().retrieval_mode,'conditional')
            core=object.__new__(Pipeline);core.s=Settings();calls=[]
            core.retriever=SimpleNamespace(prepare_dense=lambda:calls.append(1))
            core.initialize();self.assertEqual(calls,[])
            core.s.retrieval_mode='hybrid';core.initialize();self.assertEqual(calls,[1])
        with patch.dict(os.environ,{'RETRIEVAL_MODE':'hybrid'}):
            self.assertEqual(Settings().retrieval_mode,'hybrid')


if __name__=='__main__':unittest.main()
