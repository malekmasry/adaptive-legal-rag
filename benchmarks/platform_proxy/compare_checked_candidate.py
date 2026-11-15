"""Freeze and evaluate a new submission against the unchanged stress20 baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from prepare import sha,write,ROOT
from run import check,judge_pair,summarize
from services import Services
from stress_runtime import build_embeddings,make_gateway,execute

def main(a):
    start=time.monotonic();root=a.output.resolve();parent=a.parent.resolve()
    config=json.loads(a.services.read_text(encoding='utf8'))
    original=check(parent,config)
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for name in ('cases.json','baseline.json','knowledge-base.zip','controls-result.json','historical-platform.json','service-identity.json','embedding-cache.sqlite3'):
            shutil.copyfile(parent/name,root/name)
        shutil.copyfile(a.package,root/'packages'/f'{a.name}.zip')
        plan={'version':'focused-evidence-stress20','systems':[a.name], 'cases':original['cases'],
              'scope':'Development optimization of the frozen stress20; not an independent holdout.',
              'input_hashes':{n:sha(root/n) for n in ('cases.json','baseline.json','knowledge-base.zip')},
              'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},
              'embedding_code_hashes':original['embedding_code_hashes'],
              'packages':{a.name:{'sha256':sha(a.package),'top_k':40}},
              'parent_freeze_sha256':sha(parent/'freeze.json'), 'baseline_reused':True,
              'differences_from_psb':original['differences_from_psb'],
              'api_limits':{'chat_calls':110,'rerank_calls':30,'conservative_chat_token_ceiling':4000000}}
        write(root/'freeze.json',plan)
    plan=check(root,config)
    if sha(a.package)!=plan['packages'][a.name]['sha256']:raise ValueError('Candidate changed; use a new run')
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    base=json.loads((root/'baseline.json').read_text(encoding='utf8'))
    api=Services(config,root,max_chat_calls=110,max_rerank_calls=30,max_chat_tokens=4000000)
    cache=server=thread=None
    try:
        cache=build_embeddings(root,a.weights,a.seed,a.pilot,plan)
        # Local CPU startup is not an API test. Resolve exact new query vectors
        # before the product clock so cold weights cannot cause HTTP timeouts.
        sys.path.insert(0,str(ROOT.parents[1]/'focused-rag'))
        from focused_context import search_queries
        queries=[]
        for case in cases:
            query,subs=search_queries(case['question']);queries.extend([query]+subs)
        warm_start=time.monotonic();cache.embed(list(dict.fromkeys(queries)))
        write(root/'embedding-warmup.json',{'seconds':round(time.monotonic()-warm_start,3),'stats':cache.stats()})
        print(json.dumps({'embeddings_ready':True,'seconds':round(time.monotonic()-warm_start,3)}),flush=True)
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        meta=execute(root,a.name,cases,plan,f'http://127.0.0.1:{server.server_port}/v1')
        write(root/'runtime.json',{a.name:meta})
        print(json.dumps({'generated':meta}),flush=True)
        if not meta['complete']:return 2
        with (root/'generated'/a.name/'submission.csv').open(encoding='utf8',newline='') as f:
            answers={r['id']:r['answer'] for r in csv.DictReader(f)}
        write(root/'all-answers.json',{a.name:answers})
        def judge(c):
            result=judge_pair(root,api,a.name+'-'+c['id'],c['question'],c['golden'],base[c['id']]['value']['answer'],answers[c['id']])
            print(json.dumps({'judged':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':a.name,'id':c['id'],'family_id':c['family_id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(judge,cases))
        write(root/'judgments.json',results)
        report=summarize(root,plan,results,time.monotonic()-start);report['runtime']={a.name:meta}
        report['comparison_parent']=str(parent);report['spend']['includes_parent_pilot']=False
        write(root/'summary.json',report)
        print(json.dumps({'complete':True,'systems':report['systems'],'seconds':report['seconds']}),flush=True)
        return 0
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('parent','output','package','services','weights','seed'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--pilot',type=Path);p.add_argument('--name',default='focused_v1')
    raise SystemExit(main(p.parse_args()))
