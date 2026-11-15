"""Evaluate expanded hints once; reuse frozen v228/Qwen/judge inputs exactly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from prepare import ROOT,sha,write
from run import check
from run_v108_v228 import SharedRerank,answers
from compare_team_score import pair,TEAM_PROMPT
from judge import aggregate
from stress_runtime import build_embeddings,make_gateway,execute


def main(a):
    started=time.monotonic();root=a.output
    config=json.loads(a.services.read_text(encoding='utf8'));parent=check(a.parent,config);prior=check(a.prior,config)
    prompt_hash=hashlib.sha256(TEAM_PROMPT.encode()).hexdigest()
    if parent['judge_prompt_sha256']!=prompt_hash:raise ValueError('Changed judge')
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for n in ('knowledge-base.zip','cases.json','baseline.json','regression-cases.json','service-identity.json'):
            shutil.copyfile(a.parent/n,root/n)
        shutil.copyfile(a.extension_cases,root/'extension-cases.json')
        shutil.copyfile(a.prior/'embedding-cache.sqlite3',root/'embedding-cache.sqlite3')
        shutil.copyfile(a.prior/'all-answers.json',root/'extension-baseline-answers.json')
        shutil.copyfile(a.prior/'runtime.json',root/'extension-baseline-runtime.json')
        if prior['input_hashes']['extension-cases.json']!=sha(root/'extension-cases.json'):raise ValueError('Changed extension cases')
        if prior['packages']['v228_extension']['sha256']!=parent['packages']['v228']['sha256']:raise ValueError('Changed baseline runtime')
        shutil.copyfile(a.parent/'all-answers.json',root/'parent-answers.json')
        shutil.copyfile(a.parent/'judgments.json',root/'parent-judgments.json')
        shutil.copytree(a.prior/'qwen-cache',root/'qwen-cache')
        shutil.copytree(a.prior/'cache',root/'cache')
        for n,p in {'v229':a.candidate,'v229_regression':a.candidate,'v229_extension':a.candidate,'v228_extension':a.parent/'packages/v228.zip'}.items():
            shutil.copyfile(p,root/'packages'/(n+'.zip'))
        names=['knowledge-base.zip','cases.json','baseline.json','regression-cases.json','extension-cases.json','parent-answers.json','parent-judgments.json','extension-baseline-answers.json','extension-baseline-runtime.json']
        write(root/'freeze.json',{
          'version':'v229-final-source-recovery','input_hashes':{n:sha(root/n) for n in names},
          'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'embedding_code_hashes':parent['embedding_code_hashes'],
          'packages':{p.stem:{'sha256':sha(p),'top_k':40} for p in (root/'packages').glob('*.zip')},
          'judge_prompt_sha256':prompt_hash,'parent_freeze_sha256':sha(a.parent/'freeze.json'),
          'design':'Corrected v229: full42 fresh answers. Same20 baseline; same12 extension questions and saved v228 extension baseline. Prior failed candidate retained. Exact Qwen and identical full judge inputs reused; no hidden-set claim.',
          'limits':{'chat_calls':140,'rerank_calls':0,'conservative_chat_token_ceiling':8000000}})
    freeze=check(root,config)
    if sha(a.candidate)!=freeze['packages']['v229']['sha256']:raise ValueError('Changed candidate')
    cases=json.loads((root/'cases.json').read_text());reg=json.loads((root/'regression-cases.json').read_text())
    extension=json.loads((root/'extension-cases.json').read_text());base=json.loads((root/'baseline.json').read_text())
    api=SharedRerank(config,root,max_chat_calls=140,max_rerank_calls=0,max_chat_tokens=8000000)
    cache=server=thread=None
    try:
        cache=build_embeddings(root,a.weights,a.seed,None,freeze)
        sys.path.insert(0,str(a.runtime_source))
        from focused_context import search_queries
        texts=[]
        for c in cases+reg+extension:
            q,subs=search_queries(c['question']);texts.extend([q]+subs)
        warm=time.monotonic();cache.embed(list(dict.fromkeys(texts)))
        write(root/'embedding-warmup.json',{'seconds':round(time.monotonic()-warm,3),'stats':cache.stats()})
        print(json.dumps({'embeddings_warm':True,'seconds':round(time.monotonic()-warm,3)}),flush=True)
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}/v1';runtime={}
        for jobs in [[('v229',cases)],[('v229_regression',reg)],[('v229_extension',extension)]]:
            def work(job):
                name,rows=job
                result=execute(root,name,rows,freeze,url)
                print(json.dumps({'runtime':name,'seconds':result['seconds'],'counts':result.get('runtime_report',{}).get('counts')}),flush=True)
                return name,result
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                for name,value in pool.map(work,jobs):runtime[name]=value
            write(root/'runtime.json',runtime)
            if api.blocked or any(not r['complete'] or r['exit_code'] or r.get('runtime_report',{}).get('error_counts') for r in runtime.values()):
                print(json.dumps({'stopped':'runtime error/API unavailable','judging_skipped':True}),flush=True);return 2
        all_answers={name:answers(root,name) for name in runtime}
        all_answers['v228']=json.loads((root/'parent-answers.json').read_text())['v228']
        all_answers['v228_extension']=json.loads((root/'extension-baseline-answers.json').read_text())['v228_extension']
        write(root/'all-answers.json',all_answers)
        old=[r for r in json.loads((root/'parent-judgments.json').read_text()) if r['system']=='v228']
        jobs=[('v229',c,base[c['id']]['value']['answer'],all_answers['v229'][c['id']]) for c in cases]
        jobs += [('v229_vs_v228_extension',c,all_answers['v228_extension'][c['id']],all_answers['v229_extension'][c['id']]) for c in extension]
        def judge(job):
            name,c,baseline,candidate=job
            data=[c['question'],c['golden'],baseline,candidate]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            result=pair(root,api,'content-'+key,*data)
            print(json.dumps({'judged':name,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':name,'id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=old+list(pool.map(judge,jobs))
        write(root/'judgments.json',rows)
        common=set.intersection(*[{r['id'] for r in rows if r['system']==n and r['status']=='ok'} for n in ('v228','v229')])
        summary={n:{'all_evaluated':aggregate([r for r in rows if r['system']==n]),
          'common_evaluated':aggregate([r for r in rows if r['system']==n and r['id'] in common]),
          'fabrication_zero_count':sum(r.get('fabrication_zero',False) for r in rows if r['system']==n)} for n in ('v228','v229')}
        ext=[r for r in rows if r['system']=='v229_vs_v228_extension']
        done=[r for r in map(json.loads,(root/'calls.jsonl').read_text().splitlines()) if r['event']=='completed']
        usage={'deepseek_calls':sum(r['kind']=='chat' for r in done),'deepseek_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in done if r['kind']=='chat'),
          'qwen_calls':sum(r['kind']=='rerank' for r in done),'qwen_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in done if r['kind']=='rerank'),
          'qwen_cache_hits':api.cache_hits,'api_errors':sum(r.get('status')!='ok' for r in done)}
        result={'systems':summary,'extension_v229_vs_v228':aggregate(ext),'runtime':runtime,'common_ids':sorted(common),
          'seconds':round(time.monotonic()-started,3),'usage':usage,'not_platform_prediction':True,
          'timing_note':'v228 main20 timing is historical paired/shared-gateway; v229 main20 runs alone. Extra12 baseline saved from preliminary paired run. Exact Qwen cache reused. No causal standalone speed comparison.'}
        write(root/'summary.json',result);print(json.dumps(result),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('parent','prior','output','candidate','extension-cases','services','weights','seed','runtime-source'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
