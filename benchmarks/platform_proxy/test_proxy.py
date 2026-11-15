import json
from pathlib import Path
import tempfile
import unittest
import httpx

from services import Services, BudgetExceeded, validate as validate_services, DEEPSEEK, QWEN
from judge import aggregate, combine, payload, validate
from run import cached


def config():
    return {'GENERATOR_MODEL':DEEPSEEK,'JUDGE_MODEL':DEEPSEEK,'RERANKER_MODEL':QWEN,
        'GENERATOR_BASE_URL':'https://deepseek.invalid/v1','JUDGE_BASE_URL':'https://deepseek.invalid/v1',
        'RERANKER_BASE_URL':'https://qwen.invalid/v1','GENERATOR_API_KEY':'deepseek-test',
        'JUDGE_API_KEY':'deepseek-test','RERANKER_API_KEY':'qwen-test'}


def decision(score=.5, fabricated=False, extra=False):
    return {'a_score':score,'A':{'confirmed_fabrication':False,'unsupported_extra':False,'quote':'','reason':''},
        'B':{'confirmed_fabrication':fabricated,'unsupported_extra':extra,'quote':'900 рублей' if fabricated else '',
             'reason':'Неверное число' if fabricated else ''},'reason':'comparison'}


class ProxyTests(unittest.TestCase):
    def test_endpoints_credentials_models_and_paths_are_bound(self):
        seen=[]
        def handler(request):
            body=json.loads(request.content);seen.append((request.url.host,request.url.path,request.headers['Authorization'],body['model']))
            return httpx.Response(200,json={'model':body['model']})
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(handler))
            try:
                for role,model in [('GENERATOR',DEEPSEEK),('JUDGE',DEEPSEEK),('RERANKER',QWEN)]:
                    api.post(role,{'model':model})
                self.assertEqual(seen,[('deepseek.invalid','/v1/chat/completions','Bearer deepseek-test',DEEPSEEK),
                    ('deepseek.invalid','/v1/chat/completions','Bearer deepseek-test',DEEPSEEK),
                    ('qwen.invalid','/v1/rerank','Bearer qwen-test',QWEN)])
                for role,model in [('GENERATOR',QWEN),('JUDGE',QWEN),('RERANKER',DEEPSEEK),('EMBEDDER',DEEPSEEK)]:
                    with self.assertRaises(ValueError):api.post(role,{'model':model})
                self.assertEqual(len(seen),3)
                ledger=Path(directory,'calls.jsonl').read_text(encoding='utf8')
                self.assertNotIn('deepseek-test',ledger);self.assertNotIn('qwen-test',ledger)
            finally:api.close()

    def test_judge_cannot_use_reranker_account(self):
        for key,value in [('JUDGE_MODEL',QWEN),('JUDGE_BASE_URL','https://qwen.invalid/v1'),
            ('JUDGE_API_KEY','qwen-test'),('RERANKER_API_KEY','deepseek-test')]:
            bad=config();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_services(bad)

    def test_redirect_does_not_forward_credential(self):
        seen=[]
        def handler(request):seen.append(request);return httpx.Response(307,headers={'location':'https://qwen.invalid/v1/chat/completions'})
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(handler))
            try:
                with self.assertRaises(RuntimeError):api.post('JUDGE',{'model':DEEPSEEK})
                self.assertEqual(len(seen),1)
            finally:api.close()

    def test_budget_survives_resume_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(lambda r:httpx.Response(402)),max_chat_calls=1)
            try:
                with self.assertRaises(RuntimeError):api.post('JUDGE',{'model':DEEPSEEK})
            finally:api.close()
            api=Services(config(),directory,transport=httpx.MockTransport(lambda r:self.fail('Must not call')),max_chat_calls=1)
            try:
                with self.assertRaises(BudgetExceeded):api.post('JUDGE',{'model':DEEPSEEK})
            finally:api.close()

    def test_token_limit_prevents_request(self):
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(lambda r:self.fail('Must not call')),max_chat_tokens=1)
            try:
                with self.assertRaises(BudgetExceeded):api.post('JUDGE',{'model':DEEPSEEK})
            finally:api.close()

    def test_model_identity_response_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'model':QWEN})))
            try:
                with self.assertRaises(RuntimeError):api.post('JUDGE',{'model':DEEPSEEK})
            finally:api.close()

    def test_score_order_normalization_and_zero(self):
        rows=[{'status':'ok','decision':decision(.25)},{'status':'ok','decision':decision(.75)}]
        self.assertEqual(combine(rows)['score'],.75)
        rows[0]['decision']=decision(.25,True)
        self.assertEqual(combine(rows)['score'],0)

    def test_extra_fact_stress_is_separate_from_faq_score(self):
        rows=[{'status':'ok','decision':decision(.5,extra=True)},{'status':'ok','decision':decision(.5)}]
        value=combine(rows)
        self.assertEqual(value['score'],.5);self.assertEqual(value['literal_stress_score'],0)

    def test_judge_error_is_missing_not_zero_or_tie(self):
        self.assertIsNone(combine([{'status':'error'},{'status':'ok'}])['score'])
        result=aggregate([{'status':'ok','score':1},{'status':'judge_error','score':None}])
        self.assertEqual(result['score'],100);self.assertEqual(result['coverage'],.5)

    def test_arithmetic_ties_win_rate_and_mixed_scores(self):
        self.assertEqual(aggregate([{'status':'ok','score':.5}])['score'],50)
        result=aggregate([{'status':'ok','score':x} for x in [.75,.5,0]])
        self.assertAlmostEqual(result['score'],100*(.7*(1.25/3)+.3*.5))

    def test_accusation_requires_exact_quote(self):
        self.assertEqual(validate(decision(.25,True),'450 рублей','900 рублей')['a_score'],.25)
        with self.assertRaises(ValueError):validate(decision(.25,True),'450 рублей','450 рублей')

    def test_judge_payload_is_anonymous_and_ordered(self):
        a=payload('question','gold','baseline','candidate',0);b=payload('question','gold','baseline','candidate',1)
        self.assertEqual(set(a),{'question','golden','A','B'})
        self.assertEqual(a['A'],b['B']);self.assertEqual(a['B'],b['A'])

    def test_cached_failure_does_not_spend_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def fail():raise RuntimeError('temporary')
            first=cached(root,'test',fail)
            self.assertEqual(cached(root,'test',lambda:self.fail('Retry not authorized')),first)


if __name__=='__main__':unittest.main()
