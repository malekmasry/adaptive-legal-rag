"""Frozen all-three-features candidate against immutable v239 outputs.

Uses the original role firewall, exact-request caches and unmodified judge.
The observer only resolves the new citation IDs when recording initial drafts.
"""
import argparse,hashlib,json,shutil,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import run_v240_experiment as base
W=Path(__file__).resolve().parent;ROOT=W/'v242-experiment';OLD=W/'v240-experiment'
base.ROOT=ROOT;base.W=ROOT/'harness'
read=base.read;write=base.write;sha=base.sha

def freeze():
    source=read(ROOT/'source-freeze.json');original=read(OLD/'runtime-freeze.json')
    assert sha(ROOT/'packages/v242.zip')==source['package_sha256']
    plan={**original,'packages':{'v242.zip':{'sha256':source['package_sha256']}},
        'worker_sha256':sha(W/'v242_capture_worker.py'),'runner_sha256':sha(__file__),
        'source_freeze_sha256':sha(ROOT/'source-freeze.json'),
        'baseline_hashes':{p:sha(ROOT/('baseline-'+p+'.json')) for p in ('development','holdout')},
        'original_executor_sha256':sha(W/'run_v240_experiment.py')}
    file=ROOT/'runtime-freeze.json'
    if file.exists():assert read(file)==plan
    else:
        assert not (ROOT/'calls.jsonl').exists()
        base.W.mkdir(exist_ok=True)
        shutil.copyfile(W/'v242_capture_worker.py',base.W/'v240_capture_worker.py')
        for directory in ['generation-cache','cache']:
            if (OLD/directory).exists():shutil.copytree(OLD/directory,ROOT/directory)
        cache_proof={}
        for directory in ['qwen-cache','generation-cache','cache']:
            cache_proof[directory]={p.name:sha(p) for p in (ROOT/directory).glob('*.json')}
        write(ROOT/'imported-cache-hashes.json',cache_proof)
        write(file,plan)
    assert sha(base.W/'v240_capture_worker.py')==plan['worker_sha256']
    for name,digest in original['code_hashes'].items():assert sha(base.CODE/name)==digest
    for phase in ['development','holdout']:assert sha(ROOT/(phase+'.json'))==source[phase+'_sha256']
    return plan

def main(a):
    plan=freeze();source=read(ROOT/'source-freeze.json');phase=a.phase
    if phase=='holdout':
        selection=read(ROOT/'selection.json')
        assert selection['runtime_freeze_sha256']==sha(ROOT/'runtime-freeze.json')
    assert not (ROOT/(phase+'-summary.json')).exists(),'Completed phase must not be rerolled'
    start=time.monotonic();cases=read(ROOT/(phase+'.json'));baseline=read(ROOT/('baseline-'+phase+'.json'))
    assert set(baseline)=={c['id'] for c in cases}
    config=read(W/'platform-proxy-services.private.json')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==read(W/'v230-comparison/service-identity.json')['sha256']
    limits=source['protocol']['limits']
    api=base.ExactGenerationCache(config,ROOT,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],max_chat_tokens=limits['conservative_chat_byte_token_reservation'])
    server=cache=thread=None
    try:
        cache=base.build_embeddings(ROOT,W/'bge-benchmark-weights',W/'official-warm-seed',None,plan)
        server=base.make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        name='v242_'+phase;api.phase=name;url=f'http://127.0.0.1:{server.server_port}/v1'
        runtime=base.execute(name,cases,plan,url)
        answers=base.read_answers(name);covers=base.coverage(name,cases)
        details=[]
        for c in cases:
            captured=read(ROOT/'generated'/name/'captured'/(c['id']+'.json'))
            trace=(captured.get('result') or {}).get('trace',{})
            details.append({'id':c['id'],'question_brief':trace.get('question_brief'),
                'final_context_tokens':trace.get('budget',{}).get('total_evidence_tokens'),
                'schemas':[r['schema'] for r in captured['calls']],
                'status':(captured.get('result') or {}).get('status'),
                'review':trace.get('review'),'retrieval_degraded':trace.get('retrieval',{}).get('degraded',[])})
        write(ROOT/(phase+'-answers.json'),{'v239_'+phase:baseline,name:answers})
        write(ROOT/(phase+'-runtime.json'),runtime);write(ROOT/(phase+'-coverage.json'),covers)
        write(ROOT/(phase+'-feature-details.json'),details)
        print(json.dumps({'runtime':name,'seconds':runtime['seconds'],'counts':runtime['runtime_report'].get('counts'),'usage':base.usage()}),flush=True)
        if api.blocked or not runtime['complete'] or runtime['exit_code'] or runtime['runtime_report'].get('error_counts'):return 2
        api.phase=phase+'-judging'
        def judge(c):
            data=[c['question'],c['golden'],baseline[c['id']],answers[c['id']]]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            result=base.pair(ROOT,api,'content-'+key,*data)
            print(json.dumps({'judged':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':'v242_vs_v239','id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(judge,cases))
        write(ROOT/(phase+'-judgments.json'),rows)
        summary={'phase':phase,'cases':len(cases),'aggregate':base.aggregate(rows),'runtime':runtime,
            'seconds':round(time.monotonic()-start,3),'cumulative_usage':base.usage(),
            'qwen_cache_hits_phase':api.cache_hits,'generation_cache_hits_phase':api.gen_hits,'embedding_stats':cache.stats(),
            'scope':'All-three-features frozen v242 versus original immutable v239 outputs. Same corpus and unchanged judge; not platform prediction. Existing 12 holdout outputs were not inspected before source freeze. No product code changes after results.'}
        write(ROOT/(phase+'-summary.json'),summary);print(json.dumps(summary),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','holdout'],required=True)
    raise SystemExit(main(p.parse_args()))
