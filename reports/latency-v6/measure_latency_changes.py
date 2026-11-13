"""Offline measurements only. Never loads credentials or contacts a model."""
import sys,json,zipfile,tempfile,importlib.util,io
from pathlib import Path
from contextlib import redirect_stderr
from collections import defaultdict
import yaml

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/psb-rag'
sys.path.insert(0,str(SRC))
from reasoning import direct_plan,parse_history,PLANNER
from schemas import CompactPlan,Draft,Verdict
from model_api import schema_text
from security import TRUST
from token_budget import estimate_tokens
from pipeline import Pipeline
from settings import Settings
from test_pipeline import ScriptedAPI

baseline_dir=ROOT/'work/latency-baseline';baseline_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'outputs/PSB_RAG_SUBMISSION_v5.zip') as archive:
    old_prompts=yaml.safe_load(archive.read('prompts.yaml'))
    for name in ('schemas.py','pipeline.py'):
        (baseline_dir/name).write_bytes(archive.read(name))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value
old_schemas=module('baseline_schemas',baseline_dir/'schemas.py')
OldPlan=old_schemas.Plan
OldPipeline=module('baseline_pipeline',baseline_dir/'pipeline.py').Pipeline
old_schema=json.dumps(OldPlan.model_json_schema(),ensure_ascii=False)
report=json.loads((ROOT/'outputs/PSB_RELIABILITY_FULL_REPORT.json').read_text(encoding='utf8'))
rows=[]
for row in report['results']:
    q,history=parse_history(row['question'])
    payload={'question':q,'history':history,'summary':'','slots':{},'current_date':'2026-11-13'}
    def size(prompt,schema):
        messages=[{'role':'system','content':TRUST+'\n'+prompt+'\nJSON schema:\n'+schema},
                  {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        return estimate_tokens(json.dumps(messages,ensure_ascii=False),margin=1)
    fast=direct_plan(q,history,'',{},row['as_of']) is not None
    rows.append({'id':row['id'],'direct':fast,'old_input_tokens':size(old_prompts['planner'],old_schema),
                 'new_input_tokens':0 if fast else size(PLANNER,schema_text(CompactPlan)),
                 'compact_input_tokens_if_called':size(PLANNER,schema_text(CompactPlan))})

events=[json.loads(line) for line in (ROOT/'work/reliable-runtime/cache/diagnostics.jsonl').read_text(encoding='utf8').splitlines()]
events=events[max(i for i,e in enumerate(events) if e['event']=='run_configuration'):]
calls=[e for e in events if e['event']=='api_call']
fast_ids={r['id'] for r in rows if r['direct']}
by_request=defaultdict(list)
for event in calls:by_request[event['request_id']].append(event)
extra=[]
for request,items in by_request.items():
    for stage in ('Draft','Verdict'):
        extra.extend(e for e in items if e['stage']==stage)
        first=next((e for e in items if e['stage']==stage),None)
        if first is not None:extra.remove(first)

class CoveredExtra(ScriptedAPI):
    def structured(self,instruction,payload,schema):
        value=super().structured(instruction,payload,schema)
        if schema is Draft:
            extra=value.claims[0].model_copy(deep=True);extra.text='Unsupported extra assertion'
            value.claims.append(extra)
        return value

fixtures=[]
for name,factory,q in [
    ('simple_supported',lambda:ScriptedAPI(),'Какая арендная плата по А-08/2024?'),
    ('missing_information',lambda:ScriptedAPI(partial=True),'Какая арендная плата по А-08/2024 и размер пособия?'),
    ('complete_after_rejecting_extra',CoveredExtra,'Какая арендная плата по А-08/2024?'),
    ('wrong_conclusion_requires_repair',lambda:ScriptedAPI(wrong=True),'Какая арендная плата по А-08/2024?')]:
    values={}
    for label,cls in [('before',OldPipeline),('after',Pipeline)]:
        with tempfile.TemporaryDirectory() as directory,redirect_stderr(io.StringIO()):
            api=factory();core=cls(Settings(data_dir=SRC/'data',cache_dir=Path(directory),retrieval_mode='bm25',use_rerank=False),api)
            try:result=core.ask(q)
            finally:core.close()
        values[label]={'model_calls':len(api.calls),'stages':[n for n,p in api.calls],
                       'status':result['status'],'answer':result['answer']}
    fixtures.append({'scenario':name,**values})

old=sum(r['old_input_tokens'] for r in rows);new=sum(r['new_input_tokens'] for r in rows)
old_retained=sum(r['old_input_tokens'] for r in rows if not r['direct'])
new_retained=sum(r['new_input_tokens'] for r in rows if not r['direct'])
output={
    'scope':'Offline routing and scripted old/new pipeline comparison, NOT a timed live model run.',
    'external_api_calls':0,'baseline_wall_seconds':report['wall_seconds'],
    'questions':len(rows),'direct_plan_ids':sorted(fast_ids),
    'planner_calls_before':len(rows),'planner_calls_after':len(rows)-len(fast_ids),
    'planner_input_estimated_tokens_before':old,'planner_input_estimated_tokens_after':new,
    'planner_input_total_reduction_percent':round(100*(1-new/old),2),
    'retained_planner_input_reduction_percent':round(100*(1-new_retained/old_retained),2),
    'verifier_static_tokens_before':estimate_tokens(TRUST+'\n'+old_prompts['verifier']+'\nJSON schema:\n'+json.dumps(old_schemas.Verdict.model_json_schema(),ensure_ascii=False),margin=1),
    'verifier_static_tokens_after':estimate_tokens(TRUST+'\n'+__import__('prompt_config').PROMPTS['verifier']+'\nJSON schema:\n'+schema_text(Verdict),margin=1),
    'token_estimator':'cl100k approximation, margin=1; not actual DeepSeek usage',
    'removed_planners_historical_sum_seconds':round(sum(e['seconds'] for e in calls if e['stage']=='Plan' and e['request_id'] in fast_ids),3),
    'baseline_total_model_calls':len(calls),
    'baseline_additional_draft_verdict_calls':len(extra),
    'baseline_additional_calls_sum_seconds':round(sum(e['seconds'] for e in extra),3),
    'additional_call_note':'Includes repeated corrections AND format retries. Some remain necessary. First-pass verdicts were not retained, so actual eliminated corrections on the live set cannot be reconstructed.',
    'timing_note':'Historical API durations overlap across workers; sums are NOT wall-clock savings. No claim about final platform duration.',
    'fixtures':fixtures,'planner_rows':rows}
destination=ROOT/'outputs/PSB_LATENCY_V6_REPORT.json'
destination.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in output.items() if k not in ('fixtures','planner_rows')},ensure_ascii=False))
print(json.dumps([{'scenario':f['scenario'],'before':f['before']['model_calls'],'after':f['after']['model_calls']} for f in fixtures]))
