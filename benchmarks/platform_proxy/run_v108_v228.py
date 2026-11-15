"""Paired live runtimes with identical inputs, cached Qwen and team-score judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import numpy as np
from prepare import ROOT,sha,write
from run import check
from compare_team_score import pair,TEAM_PROMPT
from judge import aggregate
from resume_checked_candidate import CircuitServices
from stress_runtime import build_embeddings,make_gateway,execute


class SharedRerank(CircuitServices):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.cache_lock=threading.Lock();self.key_locks={};self.cache_hits=0
        self.cache_folder=self.directory/'qwen-cache';self.cache_folder.mkdir(exist_ok=True)
    def post(self,role,payload):
        if role!='RERANKER':return super().post(role,payload)
        if payload.get('model')!=self.config['RERANKER_MODEL']:raise ValueError('Wrong model')
        key=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.cache_lock:lock=self.key_locks.setdefault(key,threading.Lock())
        with lock:
            file=self.cache_folder/(key+'.json')
            if file.exists():
                with self.cache_lock:self.cache_hits+=1
                return json.loads(file.read_text(encoding='utf8'))
            result=super().post(role,payload);write(file,result);return result


def answers(root,name):
    with (root/'generated'/name/'submission.csv').open(encoding='utf8',newline='') as f:
        return {r['id']:r['answer'] for r in csv.DictReader(f)}


def main(a):
    started=time.monotonic();root=a.output;config=json.loads(a.services.read_text(encoding='utf8'))
    parent=check(a.parent,config);check(a.regression_parent,config)
    names=['v108','v228'];packages={'v108':a.v108,'v228':a.v228}
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for n in ('knowledge-base.zip','cases.json','baseline.json','embedding-cache.sqlite3','service-identity.json'):
            shutil.copyfile(a.parent/n,root/n)
        shutil.copyfile(a.regression_parent/'cases.json',root/'regression-cases.json')
        for name,path in packages.items():
            shutil.copyfile(path,root/'packages'/f'{name}.zip')
            shutil.copyfile(path,root/'packages'/f'{name}_regression.zip')
        write(root/'freeze.json',{'version':'v108-v228-controlled-comparison',
          'input_hashes':{n:sha(root/n) for n in ('knowledge-base.zip','cases.json','baseline.json','regression-cases.json')},
          'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},
          'embedding_code_hashes':parent['embedding_code_hashes'],
          'packages':{n:{'sha256':sha(p),'top_k':40} for name,p in packages.items() for n in (name,name+'_regression')},
          'judge_prompt_sha256':hashlib.sha256(TEAM_PROMPT.encode()).hexdigest(),
          'design':'Both variants against the same saved baseline, same 20 known development cases; additional10 manual checks. Qwen exact request cache shared; identical answer judgments reused.',
          'api_limits':{'chat_calls':200,'rerank_calls':35,'conservative_chat_token_ceiling':8000000}})
    freeze=check(root,config)
    for name,p in packages.items():
        if sha(p)!=freeze['packages'][name]['sha256']:raise ValueError('Changed runtime')
    cases=json.loads((root/'cases.json').read_text());reg=json.loads((root/'regression-cases.json').read_text())
    base=json.loads((root/'baseline.json').read_text());api=SharedRerank(config,root,max_chat_calls=200,max_rerank_calls=35,max_chat_tokens=8000000)
    cache=server=thread=None
    try:
        cache=build_embeddings(root,a.weights,a.seed,None,freeze)
        from embedding_cache import EmbeddingCache
        extra=EmbeddingCache(a.seed/'bge_embeddings.npz',a.seed/'embedding_cache.json',a.regression_parent/'embedding-cache.sqlite3')
        for key,vector in extra.vectors.items():
            if key in cache.vectors and not np.allclose(cache.vectors[key],vector,atol=1e-5):raise ValueError('Conflicting genuine embeddings')
            cache.vectors[key]=vector
        extra.close()
        sys.path.insert(0,str(ROOT.parents[1]/'focused-rag'))
        from focused_context import search_queries
        texts=[]
        for c in cases+reg:
            q,subs=search_queries(c['question']);texts.extend([q]+subs)
        warm=time.monotonic();cache.embed(list(dict.fromkeys(texts)))
        write(root/'embedding-warmup.json',{'seconds':time.monotonic()-warm,'stats':cache.stats()})
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}/v1';runtime={}
        for suffix,rows in [('',cases),('_regression',reg)]:
            def run(name):return name+suffix,execute(root,name+suffix,rows,freeze,url)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for name,meta in pool.map(run,names):
                    runtime[name]=meta;write(root/'runtime.json',runtime)
                    print(json.dumps({'runtime':name,'seconds':meta['seconds'],'counts':meta.get('runtime_report',{}).get('counts')}),flush=True)
            if api.blocked or any(not r['complete'] or r.get('runtime_report',{}).get('error_counts') for r in runtime.values()):
                print(json.dumps({'stopped':'runtime error or unavailable API','judging_skipped':True}),flush=True);return 2
        all_answers={name:answers(root,name) for name in names};write(root/'all-answers.json',all_answers)
        rows=[];locks={};lock=threading.Lock()
        def judge(job):
            name,c=job
            data=[c['question'],c['golden'],base[c['id']]['value']['answer'],all_answers[name][c['id']]]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            with lock:item_lock=locks.setdefault(key,threading.Lock())
            with item_lock:result=pair(root,api,'content-'+key,*data)
            print(json.dumps({'judged':name,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':name,'id':c['id'],**result}
        jobs=[(name,c) for c in cases for name in names]
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(judge,jobs))
        write(root/'judgments.json',rows)
        common=set.intersection(*[{r['id'] for r in rows if r['system']==n and r['status']=='ok'} for n in names])
        summary={n:{'all_evaluated':aggregate([r for r in rows if r['system']==n]),
          'common_evaluated':aggregate([r for r in rows if r['system']==n and r['id'] in common]),
          'fabrication_zero_count':sum(r.get('fabrication_zero',False) for r in rows if r['system']==n)} for n in names}
        calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
        done=[r for r in calls if r['event']=='completed']
        usage={'deepseek_calls':sum(r['kind']=='chat' for r in done),'deepseek_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in done if r['kind']=='chat'),
          'qwen_calls':sum(r['kind']=='rerank' for r in done),'qwen_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in done if r['kind']=='rerank'),
          'qwen_cache_hits':api.cache_hits,'api_errors':sum(r.get('status')!='ok' for r in done)}
        result={'systems':summary,'runtime':runtime,'common_ids':sorted(common),'seconds':round(time.monotonic()-started,3),
          'usage':usage,'not_platform_prediction':True,'timing_note':'Paired runs share a three-call gateway and exact Qwen cache; timings include contention/cache reuse and are not standalone platform estimates.'}
        write(root/'summary.json',result);print(json.dumps(result),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('parent','regression-parent','output','v108','v228','services','weights','seed'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
