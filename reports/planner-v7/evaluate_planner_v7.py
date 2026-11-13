"""Offline v6/v7 routing and retrieval replay. No credentials or external API."""
import sys,json,zipfile,importlib.util,tempfile,shutil,io
from pathlib import Path
from collections import Counter
from contextlib import redirect_stderr
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'outputs/psb-rag'
sys.path.insert(0,str(SRC))
from pipeline import Pipeline
from reasoning import direct_plan,planning_requirement,parse_history,PLANNER
from schemas import CompactPlan,Draft
from model_api import ModelError,schema_text
from security import TRUST
from token_budget import estimate_tokens
from settings import Settings

baseline_dir=ROOT/'work/planner-v6-baseline';baseline_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'outputs/PSB_RAG_SUBMISSION_v6.zip') as archive:
    (baseline_dir/'reasoning.py').write_bytes(archive.read('reasoning.py'))
spec=importlib.util.spec_from_file_location('v6_reasoning',baseline_dir/'reasoning.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
saved=json.loads((ROOT/'outputs/PSB_RELIABILITY_FULL_REPORT.json').read_text(encoding='utf8'))
with np.load(SRC/'psb_bge_fresh.npz',allow_pickle=False) as z:
    vectors=dict(zip(z['queries'].tolist(),z['query_vectors']))

class ReplayAPI:
    row=None;payload=None
    def structured(self,instruction,payload,schema):
        if schema is CompactPlan:return CompactPlan.model_validate(self.row['trace']['plan'])
        if schema is Draft:
            self.payload=payload
            # Empty stub used only to observe actual retrieval inputs, not an answer.
            return Draft(status='not_in_base')
        raise AssertionError('Unexpected model stage: '+schema.__name__)
    def embed(self,texts):
        if all(t in vectors for t in texts):return np.asarray([vectors[t] for t in texts],dtype=np.float32)
        raise ModelError('Offline query embedding not available')
    def rerank(self,*args):raise AssertionError('External reranking is prohibited')

def norm(s):return ' '.join(s.lower().replace('ё','е').replace('_',' ').split())
def covers(requirement,item):
    doc,loc=map(norm,requirement.split('|',1))
    return doc in norm(item['document']) and (not loc or loc==norm(item['locator']))
def missing(expected,items):return [r for r in expected if not any(covers(r,item) for item in items)]

rows=[]
with tempfile.TemporaryDirectory() as directory:
    cache=Path(directory)
    for vector in (SRC/'cache').glob('vectors-*.npz'):shutil.copyfile(vector,cache/vector.name)
    settings=Settings(data_dir=SRC/'data',cache_dir=cache,retrieval_mode='hybrid',use_rerank=False,as_of='2026-11-13')
    api=ReplayAPI()
    with patch('socket.socket.connect',side_effect=AssertionError('External network prohibited')),redirect_stderr(io.StringIO()):
        core=Pipeline(settings,api)
        try:
            assert core.retriever.vectors is not None
            for row in saved['results']:
                latest,history=parse_history(row['question'])
                old_direct=old.direct_plan(latest,history,'',{},settings.as_of) is not None
                new_direct=direct_plan(latest,history,'',{},settings.as_of) is not None
                payload={'question':latest,'history':history,'summary':'','slots':{},'current_date':settings.as_of}
                request_tokens=estimate_tokens(json.dumps([
                    {'role':'system','content':TRUST+'\n'+PLANNER+'\nJSON schema:\n'+schema_text(CompactPlan)},
                    {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],ensure_ascii=False),margin=1)
                entry={'id':row['id'],'question':row['question'],'v6_direct':old_direct,'v7_direct':new_direct,
                       'reason':planning_requirement(latest,history,'',{}) or 'standalone_question',
                       'v6_planner_tokens':0 if old_direct else request_tokens,
                       'v7_planner_tokens':0 if new_direct else request_tokens}
                for label,route in [('v6',old.direct_plan),('v7',direct_plan)]:
                    api.row=row;api.payload=None
                    with patch('pipeline.direct_plan',route):result=core.ask(row['question'])
                    assert result['status']!='technical_error',(row['id'],label,result['trace'])
                    plan=result['trace']['plan']
                    candidates=[{'document':Path(core.docs[core.chunks[c].doc_id].filename).stem.split('__')[-1],
                                 'locator':core.chunks[c].locator} for c in result['trace'].get('retrieval',{}).get('candidates',[])]
                    evidence=api.payload['evidence'] if api.payload else []
                    entry[label]={'as_of':plan['as_of'],'date_role':plan['date_role'],
                                  'subquestions':plan['subquestions'],'question':plan['question'],
                                  'expected':row['expected'],'candidate_missing':missing(row['expected'],candidates),
                                  'context_missing_without_reranker':missing(row['expected'],evidence),
                                  'missing_query_vectors':len(result['trace'].get('retrieval',{}).get('degraded',[])),
                                  'context_tokens':result['trace'].get('budget',{}).get('estimated_tokens_with_margin')}
                    if label=='v7' and new_direct:
                        assert plan['question']==latest
                    if label=='v7' and row['id']=='demo-5':
                        assert plan['as_of']=='2024-08' and plan['date_role']=='event'
                        assert any('ред 2023' in e['filename'] and e['locator']=='п. 2.4' for e in evidence)
                        assert not any('ред 2025' in e['filename'] for e in evidence)
                rows.append(entry)
        finally:core.close()

annotated=[r for r in rows if r['v7']['expected']]
old_calls=sum(not r['v6_direct'] for r in rows);new_calls=sum(not r['v7_direct'] for r in rows)
events=[json.loads(l) for l in (ROOT/'work/reliable-runtime/cache/diagnostics.jsonl').read_text(encoding='utf8').splitlines()]
events=events[max(i for i,e in enumerate(events) if e['event']=='run_configuration'):]
removed={r['id'] for r in rows if not r['v6_direct'] and r['v7_direct']}
output={'scope':'Offline routing plus actual hybrid retrieval with saved BGE-M3 query vectors. No reranker or generated answers. Model-route plans replay previous saved output.',
        'external_api_calls':0,'questions':len(rows),'annotated_questions':len(annotated),
        'planner_calls_v5':58,'planner_calls_v6':old_calls,'planner_calls_v7':new_calls,
        'planner_calls_reduction_vs_v6_percent':round(100*(old_calls-new_calls)/old_calls,2),
        'planner_tokens_v6':sum(r['v6_planner_tokens'] for r in rows),'planner_tokens_v7':sum(r['v7_planner_tokens'] for r in rows),
        'model_route_reasons':dict(Counter(r['reason'] for r in rows if not r['v7_direct'])),
        'newly_removed_planner_historical_sum_seconds':round(sum(e['seconds'] for e in events if e['event']=='api_call' and e['stage']=='Plan' and e['request_id'] in removed),3),
        'candidate_all_expected_v6':sum(not r['v6']['candidate_missing'] for r in annotated),
        'candidate_all_expected_v7':sum(not r['v7']['candidate_missing'] for r in annotated),
        'candidate_regressions':[r['id'] for r in annotated if set(r['v7']['candidate_missing'])-set(r['v6']['candidate_missing'])],
        'context_all_expected_without_reranker_v6':sum(not r['v6']['context_missing_without_reranker'] for r in annotated),
        'context_all_expected_without_reranker_v7':sum(not r['v7']['context_missing_without_reranker'] for r in annotated),
        'context_regressions_without_reranker':[r['id'] for r in annotated if set(r['v7']['context_missing_without_reranker'])-set(r['v6']['context_missing_without_reranker'])],
        'limitations':['No live latency or generated answer quality measurement.','Missing cached query vectors explicitly fall back to BM25.','Reranker unavailable: candidates and no-reranker context are diagnostic only.','Historical durations overlap across workers; do not subtract their sum from wall time.'],
        'results':rows}
(ROOT/'outputs/PSB_PLANNER_V7_REPORT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in output.items() if k!='results'},ensure_ascii=False))
