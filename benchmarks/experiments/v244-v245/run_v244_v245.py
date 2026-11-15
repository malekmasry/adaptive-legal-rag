"""Two isolated candidates against immutable v243 answers; exact caches only."""
import argparse,hashlib,json,shutil,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import run_v240_experiment as base

W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment';OLD=W/'v242-experiment'
REPO=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
base.ROOT=ROOT;base.W=ROOT/'harness'
read=base.read;write=base.write;sha=base.sha
SOURCE={'v244':'ed60eabc37b4897e6cf14c96f862360e3f818fc3','v245':'d88fd4d5440c2eb33f9c982f9d2d5ee402b4c8f8'}


def prepare():
    assert not ROOT.exists(),'Experiment already exists'
    ROOT.mkdir();(ROOT/'packages').mkdir();base.W.mkdir()
    for name in ('knowledge-base.zip','embedding-cache.sqlite3'):
        shutil.copyfile(OLD/name,ROOT/name)
    cases=read(OLD/'development.json')+read(OLD/'holdout.json')
    saved=read(OLD/'final-answers.json');baseline={**saved['development'],**saved['holdout']}
    assert set(baseline)=={c['id'] for c in cases}
    write(ROOT/'cases.json',cases);write(ROOT/'baseline.json',baseline)
    for name in SOURCE:shutil.copyfile(W.parent/'outputs'/f'{name}.zip',ROOT/'packages'/f'{name}.zip')
    shutil.copyfile(W.parent/'outputs/v243-final.zip',ROOT/'packages/v243-final.zip')
    shutil.copyfile(W/'v242_capture_worker.py',base.W/'v240_capture_worker.py')
    imported={}
    for name in ('generation-cache','qwen-cache','cache'):
        shutil.copytree(OLD/name,ROOT/name)
        imported[name]={p.name:sha(p) for p in (ROOT/name).glob('*.json')}
    write(ROOT/'imported-caches.json',imported)
    for name in SOURCE:shutil.copyfile(W/f'{name}-tests.log',ROOT/f'{name}-tests.log')
    original=read(OLD/'runtime-freeze.json')
    plan={**{k:original[k] for k in ('code_hashes','embedding_code_hashes','judge_prompt_sha256','full_run_as_of')},
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'runner_sha256':sha(__file__),'executor_sha256':sha(W/'run_v240_experiment.py'),
        'worker_sha256':sha(base.W/'v240_capture_worker.py'),
        'source_commits':SOURCE,'base_source':'7bbe09200176355cff5052fcb000d24aef32073a',
        'packages':{p.name:{'sha256':sha(p)} for p in (ROOT/'packages').glob('*.zip')},
        'input_hashes':{n:sha(ROOT/n) for n in ('cases.json','baseline.json','knowledge-base.zip','imported-caches.json')},
        'tests':{'v244':92,'v245':93},
        'limits':{'chat_calls':320,'rerank_calls':72,'conservative_chat_byte_token_reservation':18000000},
        'design':'30 previously exposed questions: regression comparison, NOT a new holdout. v244 changes only conditional routing and lazy index initialization. v245 changes only full-answer review into source-anchored exact patches. Identical chunking, source enrichment, generator and question brief. No candidate changes after results. Baseline is the immutable final rendering of v243. Same unchanged two-order LLM judge; no scripted quality score or favorable rerolls.',
        'runtime_note':'Three workers; exact shared embedding/rerank/generation caches. Times are not cold platform latency comparisons. Upstream draft equality is measured, not assumed.',
        'decision':'Report both candidates and manual regressions; retain platform v243 37.5/484s as reference. Local scores do not predict platform results.'}
    write(ROOT/'freeze.json',plan)
    print(json.dumps({'prepared':str(ROOT),'cases':len(cases),'packages':plan['packages'],'limits':plan['limits']}))


def check():
    plan=read(ROOT/'freeze.json')
    assert sha(__file__)==plan['runner_sha256'] and sha(W/'run_v240_experiment.py')==plan['executor_sha256']
    assert sha(base.W/'v240_capture_worker.py')==plan['worker_sha256']
    for name,digest in plan['input_hashes'].items():assert sha(ROOT/name)==digest
    for name,row in plan['packages'].items():assert sha(ROOT/'packages'/name)==row['sha256']
    for name,digest in plan['code_hashes'].items():assert sha(base.CODE/name)==digest
    assert hashlib.sha256(base.TEAM_PROMPT.encode()).hexdigest()==plan['judge_prompt_sha256']
    return plan


def main(variant):
    plan=check();cases=read(ROOT/'cases.json');baseline=read(ROOT/'baseline.json')
    assert not (ROOT/(variant+'-summary.json')).exists(),'Completed candidate cannot be rerolled'
    config=read(W/'platform-proxy-services.private.json')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==read(W/'v230-comparison/service-identity.json')['sha256']
    limits=plan['limits']
    client=base.ExactGenerationCache(config,ROOT,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],
        max_chat_tokens=limits['conservative_chat_byte_token_reservation'])
    start=time.monotonic();cache=server=thread=None
    try:
        cache=base.build_embeddings(ROOT,W/'bge-benchmark-weights',W/'official-warm-seed',None,plan)
        server=base.make_gateway(client,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        client.phase=variant+'_regression';name=client.phase
        runtime=base.execute(name,cases,plan,f'http://127.0.0.1:{server.server_port}/v1')
        answers=base.read_answers(name);covers=base.coverage(name,cases)
        write(ROOT/(variant+'-runtime.json'),runtime);write(ROOT/(variant+'-answers.json'),answers)
        write(ROOT/(variant+'-coverage.json'),covers)
        print(json.dumps({'runtime':variant,'seconds':runtime['seconds'],'counts':runtime['runtime_report'].get('counts'),'usage':base.usage()}),flush=True)
        if client.blocked or not runtime['complete'] or runtime['exit_code'] or runtime['runtime_report'].get('error_counts'):
            return 2
        client.phase=variant+'_judging'
        def judge(c):
            data=[c['question'],c['golden'],baseline[c['id']],answers[c['id']]]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            result=base.pair(ROOT,client,'content-'+key,*data)
            print(json.dumps({'judged':variant,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':variant+'_vs_v243','id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(judge,cases))
        write(ROOT/(variant+'-judgments.json'),rows)
        summary={'variant':variant,'cases':len(cases),'aggregate':base.aggregate(rows),'runtime':runtime,
            'seconds':round(time.monotonic()-start,3),'cumulative_usage':base.usage(),
            'qwen_cache_hits':client.cache_hits,'generation_cache_hits':client.gen_hits,'embedding_stats':cache.stats(),
            'scope':'Known regression questions; paired against frozen v243 answers. Both judge orders retained. No live platform score.'}
        write(ROOT/(variant+'-summary.json'),summary);print(json.dumps(summary),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        client.close()
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','v244','v245']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:raise SystemExit(main(args.action))
