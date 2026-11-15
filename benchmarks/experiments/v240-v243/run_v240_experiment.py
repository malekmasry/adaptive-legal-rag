"""Frozen source diagnosis, context-only fix and independent no-draft ablation.

Two phases: development produces no automatic selection; holdout requires a
persisted selection and frozen packages. It never changes the original judge.
"""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v240-experiment';CODE=W/'v230-benchmark-code/benchmarks/platform_proxy'
sys.path.insert(0,str(CODE))
from services import DEEPSEEK,QWEN
from run_v108_v228 import SharedRerank
from stress_runtime import build_embeddings,make_gateway,kill_tree
from compare_team_score import pair,TEAM_PROMPT
from judge import aggregate
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

class ExactGenerationCache(SharedRerank):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.gen_guard=threading.Lock();self.gen_locks={}
        self.gen_folder=self.directory/'generation-cache';self.gen_folder.mkdir(exist_ok=True)
        self.gen_hits=0;self.phase='initialization'
    def post(self,role,payload):
        if role!='GENERATOR':return super().post(role,payload)
        if payload.get('model')!=DEEPSEEK:raise ValueError('Only DeepSeek generator allowed')
        # Exact bytes used by the original ledger, not a fuzzy/prompt-prefix cache.
        key=hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()
        with self.gen_guard:lock=self.gen_locks.setdefault(key,threading.Lock())
        with lock:
            file=self.gen_folder/(key+'.json')
            if file.exists():
                saved=read(file);assert saved['payload_sha256']==key
                with self.gen_guard:
                    self.gen_hits+=1
                    with (self.directory/'generation-cache-hits.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps({'phase':self.phase,'payload_sha256':key,'origin_phase':saved['origin_phase']})+'\n')
                return saved['response']
            result=super().post(role,payload)
            write(file,{'payload_sha256':key,'origin_phase':self.phase,'response':result})
            return result

def execute(name,cases,plan,url,oracle=False):
    directory=ROOT/'generated'/name;package=ROOT/'packages'/('v239.zip' if oracle or name.startswith('v239_') else name.split('_')[0]+'.zip')
    expected=plan['packages'][package.name]['sha256'];assert sha(package)==expected
    if directory.exists():
        meta=read(directory/'execution.json') if (directory/'execution.json').exists() else {}
        if meta.get('complete') and meta.get('answers_sha256')==sha(directory/'submission.csv'):return meta
        raise ValueError('Incomplete execution retained; refusing automatic rerun')
    directory.mkdir(parents=True)
    with zipfile.ZipFile(package) as z:
        assert all(Path(n).name==n for n in z.namelist());z.extractall(directory)
    data=directory/'data';data.mkdir();shutil.copyfile(ROOT/'knowledge-base.zip',data/'knowledge-base.zip')
    with (data/'questions.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','question']);writer.writerows((c['id'],c['question']) for c in cases)
    if oracle:write(directory/'oracle_sources.json',{c['id']:c['oracle'] for c in cases})
    env={k:v for k,v in os.environ.items() if k.upper() in ('PATH','SYSTEMROOT','WINDIR','TEMP','TMP','USERPROFILE')}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',DATA_DIR=str(data),CACHE_DIR=str(directory/'cache'),OUTPUT_CSV=str(directory/'submission.csv'),
        LLM_MODEL=DEEPSEEK,LLM_BASE_URL=url,LLM_API_KEY='local-runtime',EMB_MODEL='BAAI/bge-m3',EMBEDDER_BASE_URL=url,EMBEDDER_API_KEY='local-runtime',
        RERANKER_MODEL=QWEN,RERANKER_BASE_URL=url,RERANKER_API_KEY='local-runtime',TOP_K='40',RERANKER_TOP_N='6',MAX_WORKERS='3',MAX_TOKENS='8192',
        TEMPERATURE='0',RUN_BUDGET_SECONDS='1100',AS_OF='2026-11-15',CHUNKER='common',NO_PROXY='127.0.0.1,localhost',EXPERIMENT_MODE='oracle' if oracle else 'observe')
    assert sha(W/'v240_capture_worker.py')==plan['worker_sha256']
    start=time.monotonic();proc=None;timed=False
    try:
        with (directory/'stdout.txt').open('wb') as out,(directory/'stderr.txt').open('wb') as err:
            proc=subprocess.Popen([sys.executable,'-X','utf8',str(W/'v240_capture_worker.py')],cwd=directory,env=env,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW)
            try:code=proc.wait(timeout=1200)
            except subprocess.TimeoutExpired:timed=True;kill_tree(proc);code=proc.wait(timeout=20)
    finally:
        if proc is not None:kill_tree(proc)
    answers=read_answers(name);complete=set(answers)=={c['id'] for c in cases}
    meta={'system':name,'seconds':round(time.monotonic()-start,3),'exit_code':code,'complete':complete,'outer_timeout':timed,'rows':len(answers),
        'answers_sha256':sha(directory/'submission.csv') if (directory/'submission.csv').exists() else None,'package_sha256':expected,
        'mode':'manual_source_oracle' if oracle else 'observed_unmodified_runtime',
        'runtime_report':read(directory/'submission.report.json') if (directory/'submission.report.json').exists() else {}}
    write(directory/'execution.json',meta);return meta

def read_answers(name):
    file=ROOT/'generated'/name/'submission.csv'
    if not file.exists():return {}
    with file.open(encoding='utf8',newline='') as f:return {r['id']:r['answer'] for r in csv.DictReader(f)}

def coverage(name,cases):
    rows=[]
    for c in cases:
        file=ROOT/'generated'/name/'captured'/(c['id']+'.json')
        if not file.exists():rows.append({'id':c['id'],'missing_capture':True});continue
        captured=read(file);calls=captured['calls'];payload=calls[0]['payload'] if calls else {};ev=payload.get('evidence',[])
        checks=[]
        for s in c['support']:
            label=s['filename'].removesuffix('.txt').replace('документы компании__','').replace('законы__','')
            excerpts=[e['text'] for e in ev if e['document']==label]
            for anchor in s['anchors']:checks.append({'filename':s['filename'],'anchor_sha256':hashlib.sha256(anchor.encode()).hexdigest(),'present':any(anchor in e for e in excerpts)})
        b=payload.get('context_limits',{});result=captured.get('result') or {}
        rows.append({'id':c['id'],'anchors':checks,'all_anchors_present':all(x['present'] for x in checks) if checks else None,
            'evidence_tokens':b.get('total_evidence_tokens'),'omitted_documents':b.get('omitted_documents',[]),
            'draft_origin':result.get('trace',{}).get('draft_origin'),'review':result.get('trace',{}).get('review'),
            'initial_answer':captured.get('initial_answer'),'review_changed':captured.get('initial_answer')!=result.get('answer') if captured.get('initial_answer') else None,
            'logical_model_calls':len(calls),'captured_model_seconds':sum(x['seconds'] for x in calls)})
    return rows

def usage():
    file=ROOT/'calls.jsonl';done=[r for r in map(json.loads,file.read_text(encoding='utf8').splitlines()) if r['event']=='completed'] if file.exists() else []
    return {'deepseek_calls':sum(r['kind']=='chat' for r in done),'deepseek_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in done if r['kind']=='chat'),
        'qwen_calls':sum(r['kind']=='rerank' for r in done),'qwen_input_tokens':sum(r.get('meta',{}).get('tokens',{}).get('input_tokens',0) for r in done if r['kind']=='rerank'),
        'api_errors':sum(r.get('status')!='ok' for r in done)}

def main(a):
    start=time.monotonic();dataset=read(ROOT/'dataset-freeze.json');plan=read(ROOT/'runtime-freeze.json');config=read(W/'platform-proxy-services.private.json')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==read(W/'v230-comparison/service-identity.json')['sha256']
    for name in ['development','holdout']:assert sha(ROOT/(name+'.json'))==dataset[name+'_sha256']
    assert sha(W/'run_v240_experiment.py')==plan['runner_sha256']
    for name,h in plan['code_hashes'].items():assert sha(CODE/name)==h
    assert hashlib.sha256(TEAM_PROMPT.encode()).hexdigest()==plan['judge_prompt_sha256']
    if a.phase=='holdout':
        assert (ROOT/'selection.json').exists(),'Selection must be frozen before holdout'
        selection=read(ROOT/'selection.json');assert selection['runtime_freeze_sha256']==sha(ROOT/'runtime-freeze.json')
    cases=read(ROOT/('holdout.json' if a.phase=='holdout' else 'development.json'))
    if (ROOT/(a.phase+'-summary.json')).exists():raise ValueError('Completed phase exists; do not reroll')
    limits=dataset['protocol']['limits'];api=ExactGenerationCache(config,ROOT,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],max_chat_tokens=limits['conservative_chat_byte_token_reservation'])
    cache=server=thread=None
    try:
        cache=build_embeddings(ROOT,W/'bge-benchmark-weights',W/'official-warm-seed',None,plan)
        inspect=ROOT/'inspect-v239';inspect.mkdir(exist_ok=True)
        with zipfile.ZipFile(ROOT/'packages/v239.zip') as z:z.extractall(inspect)
        sys.path.insert(0,str(inspect))
        from focused_context import search_queries
        from retrieval import controlled_queries
        from schemas import Plan
        texts=[]
        for c in cases:
            query,subs=search_queries(c['question']);p=Plan(question=query,subquestions=subs,as_of='2026-11-15')
            texts+=controlled_queries(p,subs[-1] if subs else query)
        warm=time.monotonic();cache.embed(list(dict.fromkeys(texts)))
        write(ROOT/(a.phase+'-embedding-warmup.json'),{'seconds':time.monotonic()-warm,'stats':cache.stats()})
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}/v1'
        names=['v239','oracle','v240','v241'] if a.phase=='development' else ['v239','v240','v241']
        runtime={};answers={};covers={}
        for variant in names:
            name=variant+'_'+a.phase;api.phase=name
            runtime[name]=execute(name,cases,plan,url,oracle=variant=='oracle')
            answers[name]=read_answers(name);covers[name]=coverage(name,cases)
            write(ROOT/(a.phase+'-runtime.json'),runtime);write(ROOT/(a.phase+'-answers.json'),answers);write(ROOT/(a.phase+'-coverage.json'),covers)
            print(json.dumps({'runtime':name,**runtime[name],'usage':usage()}),flush=True)
            if api.blocked or not runtime[name]['complete'] or runtime[name]['exit_code'] or runtime[name]['runtime_report'].get('error_counts'):return 2
        api.phase=a.phase+'-judging';baseline=answers['v239_'+a.phase];jobs=[]
        for variant in names:
            if variant=='v239':continue
            for c in cases:jobs.append((variant+'_vs_v239',c,baseline[c['id']],answers[variant+'_'+a.phase][c['id']]))
        if a.phase=='development':
            for variant in ['v239','oracle']:
                for c,covered in zip(cases,covers[variant+'_'+a.phase]):
                    assert c['id']==covered['id']
                    if covered.get('review_changed') and covered.get('initial_answer'):
                        jobs.append((variant+'_review_effect',c,covered['initial_answer'],answers[variant+'_'+a.phase][c['id']]))
        locks={};guard=threading.Lock()
        def judge(job):
            name,c,base,answer=job;data=[c['question'],c['golden'],base,answer]
            key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
            with guard:lock=locks.setdefault(key,threading.Lock())
            with lock:result=pair(ROOT,api,'content-'+key,*data)
            print(json.dumps({'judged':name,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':name,'id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(judge,jobs))
        write(ROOT/(a.phase+'-judgments.json'),rows)
        summary={'phase':a.phase,'cases':len(cases),'systems':{n:aggregate([r for r in rows if r['system']==n]) for n in dict.fromkeys(r['system'] for r in rows)},
            'runtime':runtime,'seconds':round(time.monotonic()-start,3),'cumulative_usage':usage(),'qwen_cache_hits_phase':api.cache_hits,'generation_cache_hits_phase':api.gen_hits,
            'scope':'Fresh full instrumented pipeline runs with exact shared caches; manual-source oracle is diagnostic. Scores use fresh v239 baseline and cannot be compared to old20. Holdout never used to alter code/selection.'}
        write(ROOT/(a.phase+'-summary.json'),summary);print(json.dumps(summary),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','holdout'],required=True)
    raise SystemExit(main(parser.parse_args()))
