"""No external APIs: compare real v7/v8 retrieval and reproduce deadline loss."""
import sys,json,zipfile,importlib.util,tempfile,shutil,io,csv,time,threading
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stderr
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'outputs/psb-rag'
sys.path.insert(0,str(SRC))
from pipeline import Pipeline
from retrieval import Retriever,controlled_queries
from model_api import ModelError
from schemas import CompactPlan,Draft,Plan
from settings import Settings
import solution

old_dir=ROOT/'work/batch-v7-baseline';old_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'outputs/PSB_RAG_SUBMISSION_v7.zip') as archive:
    for name in ('retrieval.py','solution.py','aliases.yaml'):(old_dir/name).write_bytes(archive.read(name))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result
old_retrieval=module('v7_retrieval',old_dir/'retrieval.py')
old_solution=module('v7_solution',old_dir/'solution.py')
saved=json.loads((ROOT/'outputs/PSB_RELIABILITY_FULL_REPORT.json').read_text(encoding='utf8'))['results']
with np.load(SRC/'psb_bge_fresh.npz',allow_pickle=False) as z:
    vectors=dict(zip(z['queries'].tolist(),z['query_vectors']))

class ReplayAPI:
    row=None;payload=None
    def __init__(self):self.requests=[]
    def structured(self,instruction,payload,schema):
        if schema is CompactPlan:return CompactPlan.model_validate(self.row['trace']['plan'])
        if schema is Draft:
            self.payload=payload;return Draft(status='not_in_base')
        raise AssertionError('Unexpected stage')
    def embed(self,texts):
        self.requests.append(list(texts))
        if all(t in vectors for t in texts):return np.asarray([vectors[t] for t in texts],dtype=np.float32)
        raise ModelError('Offline BGE query missing')
    def rerank(self,*args):raise AssertionError('Live reranker prohibited')

variants={}
with tempfile.TemporaryDirectory() as directory,patch('socket.socket.connect',side_effect=AssertionError('Network prohibited')),redirect_stderr(io.StringIO()):
    for label,cls in [('v7',old_retrieval.Retriever),('v8',Retriever)]:
        cache=Path(directory)/label;cache.mkdir()
        for path in (SRC/'cache').glob('vectors-*.npz'):shutil.copyfile(path,cache/path.name)
        api=ReplayAPI();settings=Settings(data_dir=SRC/'data',cache_dir=cache,retrieval_mode='hybrid',use_rerank=False,as_of='2026-11-13')
        with patch('pipeline.Retriever',cls):core=Pipeline(settings,api)
        try:
            assert core.retriever.vectors is not None
            results=[]
            for row in saved:
                api.row=row;api.payload=None;api.requests=[]
                result=core.ask(row['question'])
                assert result['status']!='technical_error',(label,row['id'],result['trace'])
                trace=result['trace'];retrieval=trace.get('retrieval',{})
                query_count=len(controlled_queries(Plan.model_validate(trace['plan']),api.payload['question'])) if api.payload else 0
                results.append({'id':row['id'],'queries':query_count,
                    'rankings':retrieval.get('rankings',[]),'candidates':retrieval.get('candidates',[]),
                    'selected':retrieval.get('selected',[]),
                    'evidence_ids':[item['chunk_id'] for item in api.payload['evidence']] if api.payload else [],
                    'embedding_mode':retrieval.get('query_embedding_mode','sequential'),
                    'fixture_embed_calls':len(api.requests)})
            variants[label]=results
        finally:core.close()

comparisons=[]
for old,new in zip(variants['v7'],variants['v8']):
    assert old['id']==new['id']
    fields=('rankings','candidates','selected','evidence_ids')
    equal={field:old[field]==new[field] for field in fields}
    assert all(equal.values()),(old['id'],equal)
    comparisons.append({'id':old['id'],'queries':old['queries'],**equal,
                        'fixture_embed_calls_v7':old['fixture_embed_calls'],'fixture_embed_calls_v8':new['fixture_embed_calls'],
                        'embedding_mode_v8':new['embedding_mode']})

deadline_cases=[]
for label,runner in [('v7',old_solution.run),('v8',solution.run)]:
    with tempfile.TemporaryDirectory() as directory,redirect_stderr(io.StringIO()):
        root=Path(directory);input_file=root/'questions.csv';output_file=root/'answers.csv'
        with input_file.open('w',encoding='utf8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['id','question']);writer.writerows([(str(i),str(i)) for i in range(3)])
        threads=[];started=[]
        settings=Settings(cache_dir=root/'cache',deadline_seconds=.35,workers=2)
        api=SimpleNamespace(deadline=0)
        def answer(question):
            threads.append(threading.current_thread());started.append(question)
            time.sleep(max(0,api.deadline-time.monotonic())+.25)
            return {'answer':'saved '+question,'status':'answered','trace':{}}
        core=SimpleNamespace(s=settings,api=api,initialize=lambda:None,ask=answer)
        with patch.object(solution,'FINALIZE_GRACE_SECONDS',.6):result=runner(input_file,output_file,core)
        for thread in set(threads):thread.join(timeout=3)
        with output_file.open(encoding='utf8',newline='') as f:answers=list(csv.DictReader(f))
        deadline_cases.append({'version':label,'counts':result['counts'],'seconds':result['seconds'],
                               'successful_answers_in_csv':sum(row['answer'].startswith('saved ') for row in answers),
                               'rows_in_csv':len(answers),'started_questions':started,
                               'collected_during_drain':result.get('collected_during_drain',0)})
assert deadline_cases[0]['successful_answers_in_csv']==0
assert deadline_cases[1]['successful_answers_in_csv']==2
before=sum(r['queries'] for r in comparisons);after=sum(r['queries']>0 for r in comparisons)
report={'scope':'Offline v7/v8 Pipeline and retrieval replay with cached BGE-M3 vectors; no live generation or reranking. Deadline comparison uses synthetic delayed workers.',
        'external_api_calls':0,'cases':len(comparisons),'identical_retrieval_cases':len(comparisons),
        'healthy_query_embedding_requests_v7':before,'healthy_query_embedding_requests_v8':after,
        'healthy_query_embedding_request_reduction_percent':round(100*(before-after)/before,2),
        'maximum_query_batch':max(r['queries'] for r in comparisons),
        'embedding_request_note':'Counts assume a working embedding service for every text. Cached local fixture lacks some rewritten query vectors; recovery calls in that fixture are listed separately and are not a live speed measurement.',
        'deadline_policy':{'work_seconds':1100,'finalize_grace_seconds':30,'new_requests_after_work_deadline':False,'cli_exits_if_workers_still_stuck':True},
        'deadline_demo':deadline_cases,'comparisons':comparisons,
        'limitations':['No measured platform speed or score.','Offline equal rankings do not measure numeric batch variation of the remote embedder.','Results not finished by the final collection cutoff cannot be retained; they are explicitly marked deadline_inflight.','The 30-second collection window consumes part of the reserve before the 1200-second platform limit.']}
(ROOT/'outputs/PSB_BATCH_DEADLINE_V8_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},ensure_ascii=False))
