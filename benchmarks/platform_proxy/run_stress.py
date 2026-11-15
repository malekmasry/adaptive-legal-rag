"""Stress20: saved donor answers, real unchanged v8 and hybrid, simple DeepSeek judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import threading
import time

from prepare import write,sha
from run import check,calibrate,baseline,judge_pair,summarize
from services import Services
from stress_runtime import build_embeddings,make_gateway,execute


def main(a):
    start=time.monotonic();root=a.run.resolve();config=json.loads(a.services.read_text(encoding='utf8'))
    plan=check(root,config);limits=plan['api_limits']
    api=Services(config,root,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],
                 max_chat_tokens=limits['conservative_chat_token_ceiling'])
    cache=server=thread=None
    try:
        controls=calibrate(root,api)
        if not controls['passed']:print('Controls failed.',flush=True);return 2
        cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
        base=baseline(root,api,cases)
        if any(r['status']!='ok' for r in base.values()):print('Baseline incomplete.',flush=True);return 3
        print(json.dumps({'baseline_ready':len(base)}),flush=True)
        cache=build_embeddings(root,a.weights,a.seed,a.pilot,plan)
        # No remote embeddings. Loading local weights, if required, is recorded
        # separately; generated query vectors are persisted before returning.
        server=make_gateway(api,cache)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        endpoint=f'http://127.0.0.1:{server.server_port}/v1'
        def generate(name):
            meta=execute(root,name,cases,plan,endpoint)
            print(json.dumps({'generation_completed':name,'rows':meta['rows'],'seconds':meta['seconds'],
                              'counts':meta.get('runtime_report',{}).get('counts')}),flush=True)
            return name,meta
        with ThreadPoolExecutor(max_workers=2) as pool:
            runtime=dict(pool.map(generate,['batch_deadline_v8','hybrid35v8']))
        write(root/'runtime.json',{'systems':runtime,'embedding_cache':cache.stats(),
            'timing_note':'Two variants run concurrently behind three shared upstream slots; elapsed is not platform-comparable.'})
        if any(not m['complete'] for m in runtime.values()):print('Missing runtime rows; stopped.',flush=True);return 4
        answers=json.loads((root/'answers.json').read_text(encoding='utf8'))
        for name in runtime:
            with (root/'generated'/name/'submission.csv').open(encoding='utf8',newline='') as stream:
                answers[name]={r['id']:r['answer'] for r in csv.DictReader(stream)}
        snapshot={'answers_sha256':{name:hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest() for name,rows in answers.items()},
                  'freeze_sha256':sha(root/'freeze.json'),'baseline_sha256':sha(root/'baseline.json')}
        snap=root/'answer-snapshot.json'
        if snap.exists() and json.loads(snap.read_text(encoding='utf8'))!=snapshot:
            raise ValueError('Answers or baseline changed since judging')
        write(snap,snapshot);write(root/'all-answers.json',answers)
        def judge(job):
            case,name=job
            # Fresh v8/hybrid answers never reuse judgments of the earlier pilot.
            result=judge_pair(root,api,name+'-'+case['id'],case['question'],case['golden'],
                base[case['id']]['value']['answer'],answers[name][case['id']])
            print(json.dumps({'judged':name,'case':case['id'],'status':result['status']}),flush=True)
            return {'system':name,'id':case['id'],'family_id':case['family_id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=list(pool.map(judge,[(c,n) for c in cases for n in plan['systems']]))
        write(root/'judgments.json',results)
        report=summarize(root,plan,results,time.monotonic()-start)
        report['runtime']=runtime;report['timing_note']='Concurrent execution; no claim of platform latency.'
        report['spend']['includes_interrupted_parent_pilot']=True
        write(root/'summary.json',report)
        print(json.dumps({'complete':len(results),'common_cases':len(report['common_cases']),'spend':report['spend']}),flush=True)
        return 0
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','services','weights','seed','pilot'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
