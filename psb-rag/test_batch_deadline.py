"""Offline embedding contracts and actual scheduler boundary regressions."""
import csv,hashlib,json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
import solution
from corpus import ingest
from model_api import API,ModelError
from retrieval import Retriever,controlled_queries
from schemas import Plan
from settings import Settings

ROOT=Path(__file__).resolve().parent

def vector(text):
    raw=hashlib.sha256(text.encode()).digest()
    return [float(raw[i]+1) for i in range(3)]

class QueryBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.docs,cls.chunks,cls.relations=ingest(ROOT/'data')
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.calls=[];self.failure=''
        # This retained suite explicitly exercises the legacy v8 Retriever.
        self.s=Settings(cache_dir=Path(self.temp.name),base_url='https://test.invalid',api_key='stub',use_rerank=False,retrieval_mode='hybrid')
        def handler(request):
            texts=json.loads(request.content)['input'];self.calls.append(texts)
            if self.failure=='auth':return httpx.Response(401)
            if self.failure=='partial' and len(texts)>1:
                return httpx.Response(200,json={'data':[{'index':0,'embedding':vector(texts[0])}]})
            if self.failure=='partial' and texts==['неизвестный запрос']:
                return httpx.Response(400)
            # Server order deliberately differs from input order.
            return httpx.Response(200,json={'data':[{'index':i,'embedding':vector(texts[i])} for i in reversed(range(len(texts)))]})
        self.api=API(self.s,httpx.MockTransport(handler))
        self.r=Retriever(self.docs,self.chunks,self.relations,self.api,self.s)
        matrix=np.array([vector(c.id) for c in self.r.chunks],dtype=np.float32)
        self.r.vectors=matrix/np.linalg.norm(matrix,axis=1)[:,None]
        self.plan=Plan(question='Срок оплаты П-17/2024?',subquestions=['Доставка П-17/2024?','Неустойка П-17/2024?'],rewrites=['П-17/2024 условия оплаты'])
    def tearDown(self):self.api.close();self.temp.cleanup()
    def test_single_batch_preserves_exact_sequential_rankings_and_context(self):
        fn=self.r.dense_many
        with patch.object(self.r,'dense_many',side_effect=lambda qs,p:[fn([q],p)[0] for q in qs]):
            before,before_trace=self.r.search(self.plan,original_query=self.plan.question)
        queries=controlled_queries(self.plan,self.plan.question)
        self.assertEqual(len(self.calls),len(queries));self.calls.clear()
        after,after_trace=self.r.search(self.plan,original_query=self.plan.question)
        self.assertEqual(self.calls,[queries])
        self.assertEqual(after_trace['rankings'],before_trace['rankings'])
        self.assertEqual(after_trace['candidates'],before_trace['candidates'])
        self.assertEqual([c.id for c in before],[c.id for c in after])
    def test_duplicate_queries_are_embedded_once(self):
        self.plan.subquestions=[self.plan.question,self.plan.question];self.plan.rewrites=[]
        self.r.search(self.plan,original_query=self.plan.question)
        self.assertEqual(self.calls,[[self.plan.question]])
    def test_partial_batch_recovers_good_items_without_misalignment(self):
        self.failure='partial';self.plan.subquestions=['неизвестный запрос'];self.plan.rewrites=[]
        _,trace=self.r.search(self.plan,original_query=self.plan.question)
        self.assertEqual(trace['query_embedding_mode'],'per_query_recovery')
        self.assertEqual(self.calls,[[self.plan.question,'неизвестный запрос'],[self.plan.question],['неизвестный запрос']])
        self.assertEqual(sum(r['method']=='dense' for r in trace['rankings']),1)
        self.assertEqual(sum(r['method']=='bm25' for r in trace['rankings']),2)
    def test_auth_failure_does_not_fan_out(self):
        self.failure='auth';_,trace=self.r.search(self.plan,original_query=self.plan.question)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(trace['query_embedding_mode'],'unavailable')
        self.assertTrue(trace['candidates'])
    def test_expired_deadline_does_not_send_batch_or_recovery_requests(self):
        self.api.deadline=time.monotonic()-1
        _,trace=self.r.search(self.plan,original_query=self.plan.question)
        self.assertEqual(self.calls,[]);self.assertEqual(trace['query_embedding_mode'],'unavailable')
    def test_bm25_mode_does_not_request_embeddings(self):
        self.r.search(self.plan,mode='bm25');self.assertEqual(self.calls,[])

class DeadlineCollection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.input=self.root/'questions.csv';self.output=self.root/'submission.csv'
        with self.input.open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['id','question']);w.writerows([(str(i),str(i)) for i in range(3)])
        self.settings=Settings(cache_dir=self.root/'cache',deadline_seconds=.35,workers=2)
    def tearDown(self):self.temp.cleanup()
    def pipeline(self,ask,initialize=lambda:None):
        return SimpleNamespace(s=self.settings,api=SimpleNamespace(deadline=0),ask=ask,initialize=initialize)
    def answers(self):
        with self.output.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
    def test_late_successes_are_saved_and_new_questions_do_not_start(self):
        started=[]
        def ask(q):
            started.append(q)
            time.sleep(max(0,core.api.deadline-time.monotonic())+.15)
            return {'answer':'saved '+q,'status':'answered','trace':{}}
        core=self.pipeline(ask)
        with patch.object(solution,'FINALIZE_GRACE_SECONDS',.5):report=solution.run(self.input,self.output,core)
        self.assertEqual(set(started),{'0','1'})
        self.assertEqual(report['counts'],{'answered':2,'deadline_unprocessed':1})
        self.assertEqual(report['collected_during_drain'],2);self.assertEqual(report['unfinished_workers'],0)
        self.assertEqual([r['answer'] for r in self.answers()[:2]],['saved 0','saved 1'])
    def test_completed_result_is_harvested_even_if_wait_returns_after_work_deadline(self):
        self.settings.workers=1;started=[];original_wait=solution.wait
        def ask(q):
            started.append(q);return {'answer':'saved '+q,'status':'answered','trace':{}}
        core=self.pipeline(ask)
        def delayed_wait(futures,**kwargs):
            time.sleep(max(0,core.api.deadline-time.monotonic())+.05)
            return original_wait(futures,timeout=0)
        # Delay result production just enough that the scheduler enters wait().
        original_ask=core.ask
        core.ask=lambda q:(time.sleep(.05),original_ask(q))[1]
        with patch.object(solution,'FINALIZE_GRACE_SECONDS',.4),patch.object(solution,'wait',side_effect=delayed_wait):
            report=solution.run(self.input,self.output,core)
        self.assertEqual(started,['0']);self.assertEqual(self.answers()[0]['answer'],'saved 0')
        self.assertEqual(report['counts']['answered'],1)
    def test_exhausted_initialization_does_not_schedule_questions(self):
        started=[]
        core=self.pipeline(lambda q:started.append(q))
        core.initialize=lambda:time.sleep(max(0,core.api.deadline-time.monotonic())+.02)
        with patch.object(solution,'FINALIZE_GRACE_SECONDS',.1):report=solution.run(self.input,self.output,core)
        self.assertEqual(started,[]);self.assertEqual(report['counts'],{'deadline_unprocessed':3})
    def test_stuck_worker_is_bounded_without_losing_other_answers(self):
        release=threading.Event();threads=[]
        def ask(q):
            if q=='0':
                threads.append(threading.current_thread());release.wait()
            else:time.sleep(max(0,core.api.deadline-time.monotonic())+.05)
            return {'answer':'saved '+q,'status':'answered','trace':{}}
        core=self.pipeline(ask)
        try:
            with patch.object(solution,'FINALIZE_GRACE_SECONDS',.2):report=solution.run(self.input,self.output,core)
            self.assertEqual(report['statuses'],{'1':'answered','0':'deadline_inflight','2':'deadline_unprocessed'})
            self.assertEqual(report['unfinished_workers'],1)
            self.assertEqual(self.answers()[1]['answer'],'saved 1')
        finally:
            release.set()
            for thread in threads:thread.join(timeout=3)
    def test_cli_exits_after_saving_even_with_a_permanently_stuck_thread(self):
        code='''import csv,json,threading
from pathlib import Path
from types import SimpleNamespace
from settings import Settings
import solution
root=Path.cwd()
s=Settings(cache_dir=root/'cache-child',deadline_seconds=.2,workers=1)
core=SimpleNamespace(s=s,api=SimpleNamespace(deadline=0),initialize=lambda:None,
    ask=lambda q:threading.Event().wait())
solution.FINALIZE_GRACE_SECONDS=.2
solution.finish_cli(solution.run(root/'questions.csv',root/'child.csv',core))
'''
        env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONIOENCODING':'utf-8'}
        child=subprocess.run([sys.executable,'-c',code],cwd=self.root,env=env,text=True,capture_output=True,timeout=120)
        self.assertEqual(child.returncode,0,child.stderr)
        report=json.loads(child.stdout);self.assertEqual(report['unfinished_workers'],1)
        self.assertTrue((self.root/'child.report.json').exists())
        with (self.root/'child.csv').open(encoding='utf8',newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),3)

if __name__=='__main__':unittest.main()
