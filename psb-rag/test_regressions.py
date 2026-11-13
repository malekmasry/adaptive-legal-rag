"""Regression tests for failures observed in real model runs, plus retrieval boundaries."""
import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import httpx,numpy as np
from corpus import ingest
from settings import Settings
from retrieval import Retriever,event_window,controlled_queries,resolve_entity,explicit_event_date
from reasoning import validate_claim,evidence_payload
from schemas import Plan,Claim,Citation,Calculation
from model_api import API,ModelError

ROOT=Path(__file__).resolve().parent

class Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.docs,cls.chunks,cls.relations=ingest(ROOT/'data')
        cls.s=Settings(cache_dir=Path(cls.tmp.name),use_rerank=False)
        cls.r=Retriever(cls.docs,cls.chunks,cls.relations,None,cls.s)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_explicit_approval_date_overrides_unspecified_model_role(self):
        p=Plan(question='Кто согласует договор на 12 млн рублей 28.02.2025?',as_of='2025-02-28')
        self.assertEqual(event_window(p,'2026-11-13'),('2025-02-28','2025-02-28'))
        names={self.r.docs[self.r.chunks[i].doc_id].filename for i in self.r.pool(p)}
        self.assertTrue(any('ред 2023' in n for n in names));self.assertFalse(any('ред 2025' in n for n in names))
    def test_dense_search_independent_of_zero_lexical_scores(self):
        chosen=self.r.chunks[10].id
        with patch.object(self.r,'lexical',return_value=[]),patch.object(self.r,'dense',return_value=[(chosen,.8)]) as dense:
            chunks,trace=self.r.search(Plan(question='zzzzzz'),mode='hybrid')
        dense.assert_called_once();self.assertIn(chosen,[c.id for c in chunks])
    def test_explicit_historical_month_survives_unspecified_current_plan(self):
        p=Plan(question='Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?',
               as_of='2026-11-13',date_role='unspecified')
        self.assertEqual(event_window(p,'2026-11-13'),('2024-08-01','2024-08-31'))
        names={self.r.docs[self.r.chunks[i].doc_id].filename for i in self.r.pool(p)}
        self.assertTrue(any('ред 2023' in name for name in names))
        self.assertFalse(any('ред 2025' in name for name in names))
    def test_month_boundaries_and_iso_approval_dates(self):
        for question,expected in [
            ('Кто согласует договор в феврале 2025 года?',('2025-02-01','2025-02-28')),
            ('Кто согласует договор в марте 2025 года?',('2025-03-01','2025-03-31')),
            ('Нужно ли было одобрение в 2024 году?',('2024-01-01','2024-12-31')),
            ('Кто согласует договор на дату 2025-02-28?',('2025-02-28','2025-02-28'))]:
            with self.subTest(question=question):
                self.assertEqual(event_window(Plan(question=question),'2026-11-13'),expected)
    def test_signature_and_specification_do_not_replace_current_event(self):
        for question in ('П-17/2024: как согласовать изменения сейчас?',
                         'Договор подписан в августе 2024. Как сейчас согласовать расторжение?',
                         'На какой адрес поставлять по П-17/2024, спецификация подписана 02.10.2024?'):
            with self.subTest(question=question):
                self.assertEqual(explicit_event_date(question),'')
    def test_rewrite_cannot_switch_company_or_numbers(self):
        p=Plan(question='Оплата П-17/2024 за 30 дней?',rewrites=['Оплата ПТ-05/2023 за 45 дней?','П-17/2024 срок оплаты'])
        queries=controlled_queries(p,p.question)
        self.assertIn(p.question,queries);self.assertEqual(len(queries),2)
    def test_shared_address_is_not_unique_contract_alias(self):
        entity,candidates=resolve_entity('склад в Химках')
        self.assertEqual(entity,'');self.assertEqual(set(candidates),{'a08','p17'})
    def test_calculation_russian_units_and_thousands(self):
        base=Citation(chunk_id='a',quote='450 000 рублей в месяц')
        rate=Citation(chunk_id='b',quote='не более чем на 7%')
        evidence={'a':{'text':base.quote},'b':{'text':rate.quote}}
        for unit in ['рублей в месяц','руб./мес.','руб.']:
            claim=Claim(text='Условная плата — 481 500 рублей в месяц',citations=[base,rate],
                calculation=Calculation(expression='450000 * (1 + 7 / 100)',result='481500',unit=unit,operands=[base,rate]))
            self.assertEqual(validate_claim(claim,evidence),[])
        claim.calculation.result='481000';self.assertIn('calculation result mismatch',validate_claim(claim,evidence))
    def test_parent_is_complete_or_limitation_is_explicit(self):
        parent=next(c for c in self.chunks if c.kind=='parent' and len(c.text)>30000)
        child=next(c for c in self.chunks if c.parent_id==parent.id)
        evidence,budget=evidence_payload([child,parent],{d.id:d for d in self.docs},8000)
        self.assertEqual(len(evidence),1)
        self.assertFalse(evidence[0]['governing_context']['complete'])
        self.assertIn(evidence[0]['governing_context']['text'],parent.text)
        self.assertIn(parent.id,budget['omitted'])
    def test_loaded_bge_cache_is_normalized_and_changes_with_model(self):
        with np.load(ROOT/'psb_bge_fresh.npz',allow_pickle=False) as z:
            self.assertEqual(str(z['cache_key']),self.r.cache_key)
            self.assertEqual(z['vectors'].shape,(316,1024))
            np.testing.assert_allclose(np.linalg.norm(z['vectors'],axis=1),1,atol=1e-4)
    def test_protocol_disconnect_retried_without_raw_exception(self):
        calls=[]
        def handler(request):
            calls.append(1)
            if len(calls)<3:raise httpx.RemoteProtocolError('sensitive transport contents')
            return httpx.Response(200,json={'data':[{'index':0,'embedding':[1,0]}]})
        s=Settings(cache_dir=Path(self.tmp.name),base_url='https://test.invalid',api_key='secret')
        api=API(s,transport=httpx.MockTransport(handler))
        try:self.assertEqual(api.embed(['x']).shape,(1,2));self.assertEqual(len(calls),3)
        finally:api.close()
    def test_output_truncation_is_not_a_valid_json_answer(self):
        response={'choices':[{'finish_reason':'length','message':{'content':'{"question":"x"}'}}]}
        s=Settings(cache_dir=Path(self.tmp.name),base_url='https://test.invalid',api_key='secret',model='test')
        api=API(s,transport=httpx.MockTransport(lambda r:httpx.Response(200,json=response)))
        try:
            with self.assertRaisesRegex(ModelError,'truncated'):api.structured('x',{},Plan)
        finally:api.close()

if __name__=='__main__':unittest.main(verbosity=2)
