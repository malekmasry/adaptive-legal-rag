"""Frozen v230 comparison: old20 common baseline, old12/new20 direct vs v229."""
import argparse,hashlib,json,shutil,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare import ROOT,sha,write
from run import check
from run_v108_v228 import SharedRerank,answers
from compare_team_score import pair,TEAM_PROMPT
from judge import aggregate
from stress_runtime import build_embeddings,make_gateway,execute

def main(a):
    started=time.monotonic();root=a.output
    config=json.loads(a.services.read_text(encoding='utf8'));parent=check(a.parent,config)
    prompt_hash=hashlib.sha256(TEAM_PROMPT.encode()).hexdigest()
    assert parent['judge_prompt_sha256']==prompt_hash
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for n in ('knowledge-base.zip','cases.json','baseline.json','regression-cases.json','extension-cases.json','service-identity.json','embedding-cache.sqlite3'):
            shutil.copyfile(a.parent/n,root/n)
        shutil.copyfile(a.extra,root/'extra-cases.json')
        shutil.copyfile(a.parent/'all-answers.json',root/'parent-answers.json')
        shutil.copyfile(a.parent/'judgments.json',root/'parent-judgments.json')
        shutil.copytree(a.parent/'qwen-cache',root/'qwen-cache');shutil.copytree(a.parent/'cache',root/'cache')
        for n,p in {'v230':a.candidate,'v230_regression':a.candidate,'v230_extension':a.candidate,'v230_extra':a.candidate,'v229_extra':a.parent/'packages/v229.zip'}.items():
            shutil.copyfile(p,root/'packages'/(n+'.zip'))
        names=['knowledge-base.zip','cases.json','baseline.json','regression-cases.json','extension-cases.json','extra-cases.json','parent-answers.json','parent-judgments.json']
        write(root/'freeze.json',{'version':'v230-decision-catalog','input_hashes':{n:sha(root/n) for n in names},
          'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'embedding_code_hashes':parent['embedding_code_hashes'],
          'packages':{p.stem:{'sha256':sha(p),'top_k':40} for p in (root/'packages').glob('*.zip')},
          'judge_prompt_sha256':prompt_hash,'parent_freeze_sha256':sha(a.parent/'freeze.json'),
          'design':'62 fresh v230 answers: known20 + regression10 + known extension12 +20 newly frozen source-authored development questions. Fresh v229 only on new20; historical v229 answers retained unchanged on previous42. Old20 use saved common baseline. Extension12 and new20 compare v230 directly with v229. Exact Qwen request caching shared. No hidden-platform claim.',
          'limits':{'chat_calls':300,'rerank_calls':22,'conservative_chat_token_ceiling':18000000}})
    freeze=check(root,config)
    if sha(a.candidate)!=freeze['packages']['v230']['sha256']:raise ValueError('Changed candidate')
    read=lambda n:json.loads((root/n).read_text(encoding='utf8'))
    cases,reg,ext,new=[read(n) for n in ['cases.json','regression-cases.json','extension-cases.json','extra-cases.json']]
    base=read('baseline.json');old_answers=read('parent-answers.json')
    api=SharedRerank(config,root,max_chat_calls=300,max_rerank_calls=22,max_chat_tokens=18000000)
    cache=server=thread=None
    try:
        cache=build_embeddings(root,a.weights,a.seed,None,freeze)
        sys.path.insert(0,str(a.runtime_source))
        from focused_context import search_queries
        texts=[]
        for c in cases+reg+ext+new:
            q,subs=search_queries(c['question']);texts.extend([q]+subs)
        warm=time.monotonic();cache.embed(list(dict.fromkeys(texts)))
        write(root/'embedding-warmup.json',{'seconds':round(time.monotonic()-warm,3),'stats':cache.stats()})
        print(json.dumps({'embeddings_warm':True,'seconds':round(time.monotonic()-warm,3)}),flush=True)
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}/v1'
        runtime={}
        jobs=[('v230',cases),('v230_regression',reg),('v230_extension',ext),('v229_extra',new),('v230_extra',new)]
        for name,rows in jobs:
            result=execute(root,name,rows,freeze,url);runtime[name]=result;write(root/'runtime.json',runtime)
            print(json.dumps({'runtime':name,'seconds':result['seconds'],'counts':result.get('runtime_report',{}).get('counts')}),flush=True)
            if api.blocked or not result['complete'] or result['exit_code'] or result.get('runtime_report',{}).get('error_counts'):
                print(json.dumps({'stopped':'runtime error/API unavailable','judging_skipped':True}),flush=True);return 2
        all_answers={name:answers(root,name) for name in runtime}
        for name in ['v229','v229_regression','v229_extension']:all_answers[name]=old_answers[name]
        write(root/'all-answers.json',all_answers)
        old=[r for r in read('parent-judgments.json') if r['system']=='v229']
        jobs=[('v230',c,base[c['id']]['value']['answer'],all_answers['v230'][c['id']]) for c in cases]
        jobs += [('v230_vs_v229_extension',c,all_answers['v229_extension'][c['id']],all_answers['v230_extension'][c['id']]) for c in ext]
        jobs += [('v230_vs_v229_extra',c,all_answers['v229_extra'][c['id']],all_answers['v230_extra'][c['id']]) for c in new]
        def judge(job):
            name,c,baseline,candidate=job;data=[c['question'],c['golden'],baseline,candidate]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            result=pair(root,api,'content-'+key,*data)
            print(json.dumps({'judged':name,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':name,'id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=old+list(pool.map(judge,jobs))
        write(root/'judgments.json',rows)
        systems={n:{'aggregate':aggregate([r for r in rows if r['system']==n]),'fabrication_zeros':sum(r.get('fabrication_zero',False) for r in rows if r['system']==n)} for n in ['v229','v230']}
        direct={cohort:aggregate([r for r in rows if r['system']=='v230_vs_v229_'+cohort]) for cohort in ['extension','extra']}
        done=[r for r in map(json.loads,(root/'calls.jsonl').read_text(encoding='utf8').splitlines()) if r['event']=='completed']
        usage={'deepseek_calls':sum(r['kind']=='chat' for r in done),'deepseek_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in done if r['kind']=='chat'),
          'qwen_calls':sum(r['kind']=='rerank' for r in done),'qwen_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in done if r['kind']=='rerank'),
          'qwen_cache_hits':api.cache_hits,'api_errors':sum(r.get('status')!='ok' for r in done)}
        result={'systems':systems,'direct_comparisons':direct,'runtime':runtime,'seconds':round(time.monotonic()-started,3),'usage':usage,
          'not_platform_prediction':True,'timing_note':'All runtime workers3. Known v230 Qwen requests cached. New20 v229 incurs fresh Qwen, v230 reuses identical results. v229 historical timing and new20 timing are not a controlled speed benchmark.'}
        write(root/'summary.json',result);print(json.dumps(result),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['parent','output','candidate','extra','services','weights','seed','runtime-source']:p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
