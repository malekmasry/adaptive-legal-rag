"""Cheap, resumable local LLM comparison. No private holds are opened."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import statistics

from judge import PROMPT, GENERATOR_PROMPT, validate, combine, aggregate, payload
from prepare import sha, write
from services import Services, BudgetExceeded

ROOT=Path(__file__).resolve().parent


def check(root,config):
    freeze=json.loads((root/'freeze.json').read_text(encoding='utf8'))
    for name,digest in freeze['input_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen input changed: '+name)
    for name,digest in freeze['code_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen implementation changed: '+name)
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    file=root/'service-identity.json'
    if file.exists() and json.loads(file.read_text(encoding='utf8'))['sha256']!=identity:
        raise ValueError('Endpoint/model/credentials changed; start a separate experiment')
    if not file.exists():write(file,{'sha256':identity})
    return freeze


def cached(root,key,call):
    path=root/'cache'/f'{key}.json';path.parent.mkdir(exist_ok=True)
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    try:value={'status':'ok','value':call()}
    except BudgetExceeded:raise
    except Exception as exc:value={'status':'error','error_type':type(exc).__name__}
    # No automatic re-spending on invalid JSON, HTTP 402, or already saved errors.
    write(path,value);return value


def judge_pair(root,api,key,question,golden,baseline,candidate):
    rounds=[]
    for order in (0,1):
        data=payload(question,golden,baseline,candidate,order)
        def call(data=data):return validate(api.chat('JUDGE',PROMPT,data),data['A'],data['B'])
        result=cached(root,key+f'-{order}',call)
        rounds.append({'status':result['status'],'decision':result.get('value'),
                       'error_type':result.get('error_type')})
    return {'rounds':rounds,**combine(rounds)}


def calibrate(root,api):
    controls=json.loads((root/'controls.json').read_text(encoding='utf8'))
    def work(row):
        result=judge_pair(root,api,'control-'+row['id'],row['question'],row['golden'],row['baseline'],row['candidate'])
        score=result['score'];expect=row['expect']
        passed=result['status']=='ok' and {'tie':score==.5,'zero':score==0,
            'loss':score is not None and score<.5,'win':score is not None and score>.5}[expect]
        return {'id':row['id'],'passed':passed,**result}
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(work,controls))
    result={'passed':all(r['passed'] for r in rows),'rows':rows,'scope':'sanity controls, not platform calibration'}
    write(root/'controls-result.json',result)
    return result


def baseline(root,api,cases):
    def work(case):
        def generate():
            ranked=cached(root,'rerank-'+case['id'],lambda:api.rerank(case['query'],case['rerank_texts']))
            if ranked['status']!='ok':raise ValueError('Baseline reranking failed')
            evidence=[case['retrieval_candidates'][i] for i in ranked['value']]
            response=api.chat('GENERATOR',GENERATOR_PROMPT,{'question':case['question'],'evidence':evidence},max_tokens=1000)
            if not isinstance(response.get('answer'),str) or not response['answer'].strip():raise ValueError('Missing answer')
            return {'answer':response['answer'],'source_ids':[r['id'] for r in evidence]}
        return case['id'],cached(root,'baseline-'+case['id'],generate)
    with ThreadPoolExecutor(max_workers=2) as pool:result=dict(pool.map(work,cases))
    write(root/'baseline.json',result)
    return result


def rank_correlation(left,right):
    def ranks(items):
        return [sum(x<value for x in items)+(sum(x==value for x in items)-1)/2 for value in items]
    a,b=ranks(left),ranks(right)
    if len(a)<3 or len(set(a))<2 or len(set(b))<2:return None
    return statistics.correlation(a,b)


def summarize(root,freeze,results,seconds):
    summary={}
    common=set(freeze['cases'])
    for name in freeze['systems']:
        rows=[r for r in results if r['system']==name]
        common&={r['id'] for r in rows if r['status']=='ok'}
        summary[name]={'psb_rule_proxy':aggregate(rows),'literal_gold_stress':aggregate(rows,'literal_stress_score'),
            'fabrication_zero_count':sum(r.get('fabrication_zero',False) for r in rows),
            'unsupported_extra_count':sum(r.get('unsupported_extra',False) for r in rows),
            'mean_order_gap':statistics.mean([r['order_gap'] for r in rows if r['status']=='ok']) if any(r['status']=='ok' for r in rows) else None}
    for name in summary:
        rows=[r for r in results if r['system']==name and r['id'] in common]
        summary[name]['common_psb_rule_proxy']=aggregate(rows)
        summary[name]['common_literal_gold_stress']=aggregate(rows,'literal_stress_score')
    observed=json.loads((root/'historical-platform.json').read_text(encoding='utf8'))['observations']
    historical={r['id']:r.get('official_score',r.get('score')) for r in observed}
    names=[n for n in summary if n in historical and summary[n]['common_psb_rule_proxy']['score'] is not None]
    corr=rank_correlation([historical[n] for n in names],[summary[n]['common_psb_rule_proxy']['score'] for n in names])
    report={'version':freeze['version'],'seconds':round(seconds,3),'systems':summary,'common_cases':sorted(common),
        'historical_platform_scores':{n:historical[n] for n in names},'exploratory_rank_spearman':corr,
        'calibrated_to_psb':False,'known_scores_hidden_from_judge':True,'baseline_generated_without_golden':True,
        'limits':freeze['differences_from_psb'],'scope':freeze['scope']}
    calls=[json.loads(line) for line in (root/'calls.jsonl').read_text(encoding='utf8').splitlines()]
    completed=[x for x in calls if x['event']=='completed']
    report['spend']={'deepseek_calls':sum(x['kind']=='chat' for x in completed),
        'qwen_rerank_calls':sum(x['kind']=='rerank' for x in completed),
        'deepseek_reported_tokens':sum(x.get('usage',{}).get('total_tokens',0) for x in completed if x['kind']=='chat'),
        'qwen_reported_usage':[x.get('meta',{}) for x in completed if x['kind']=='rerank']}
    write(root/'summary.json',report)
    return report


def main(a):
    start=time.monotonic();root=a.run.resolve();config=json.loads(a.services.read_text(encoding='utf8'))
    freeze=check(root,config);limits=freeze['api_limits']
    api=Services(config,root,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],
                 max_chat_tokens=limits['conservative_chat_token_ceiling'])
    try:
        control=calibrate(root,api)
        print(json.dumps({'controls_passed':control['passed']}),flush=True)
        if not control['passed']:return 2
        cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
        answers=json.loads((root/'answers.json').read_text(encoding='utf8'))
        baselines=baseline(root,api,cases)
        if any(x['status']!='ok' for x in baselines.values()):
            print('Baseline incomplete; no comparison launched.',flush=True);return 3
        jobs=[(case,name) for case in cases for name in freeze['systems']]
        def work(job):
            case,name=job
            result=judge_pair(root,api,name+'-'+case['id'],case['question'],case['golden'],
                baselines[case['id']]['value']['answer'],answers[name][case['id']])
            print(json.dumps({'system':name,'case':case['id'],'status':result['status']}),flush=True)
            return {'system':name,'id':case['id'],'family_id':case['family_id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(work,jobs))
        write(root/'judgments.json',results)
        summary=summarize(root,freeze,results,time.monotonic()-start)
        print(json.dumps({'completed':len(results),'common_cases':len(summary['common_cases']),'spend':summary['spend']}),flush=True)
        return 0
    finally:api.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--services',type=Path,required=True);raise SystemExit(main(parser.parse_args()))
