"""Mandatory UI-label cleanup: exact development replay, then live holdout.

No selection between dirty and clean rendering. Unchanged judgments, including
unfavorable ones, are reused. Neither generation prompts nor factual text change.
"""
import argparse,ast,hashlib,json,sys,threading,time,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import run_v242_experiment as original
base=original.base;ROOT=original.ROOT;W=original.W
read=base.read;write=base.write;sha=base.sha

def frozen():
    plan=read(ROOT/'runtime-freeze.json');source=read(ROOT/'v243-source-freeze.json')
    assert sha(ROOT/'packages/v243.zip')==source['package_sha256']
    assert sha(ROOT/'packages/v242.zip')==source['prototype_package_sha256']
    with zipfile.ZipFile(ROOT/'packages/v242.zip') as old,zipfile.ZipFile(ROOT/'packages/v243.zip') as new:
        assert old.namelist()==new.namelist()
        assert [n for n in old.namelist() if old.read(n)!=new.read(n)]==['question_brief.py']
        before=ast.parse(old.read('question_brief.py'));after=ast.parse(new.read('question_brief.py'))
        # Every definition except the pure reference/text renderer is identical.
        a=[ast.dump(n) for n in before.body if not isinstance(n,ast.FunctionDef) or n.name!='resolved_paragraphs']
        b=[ast.dump(n) for n in after.body if not isinstance(n,ast.FunctionDef) or n.name!='resolved_paragraphs']
        assert a==b
    value={'source_freeze_sha256':sha(ROOT/'v243-source-freeze.json'),'prototype_runtime_freeze_sha256':sha(ROOT/'runtime-freeze.json'),
        'runner_sha256':sha(__file__),'worker_sha256':sha(base.W/'v240_capture_worker.py'),
        'baseline_hashes':plan['baseline_hashes'],'judge_prompt_sha256':hashlib.sha256(base.TEAM_PROMPT.encode()).hexdigest(),
        'scope':'Only own-part leading UI labels are removed. No model prompt/input changes; no factual optimization after seeing results.'}
    file=ROOT/'v243-evaluation-freeze.json'
    if file.exists():assert read(file)==value
    else:write(file,value)
    return {**plan,'packages':{'v243.zip':{'sha256':source['package_sha256']}}}

def api():
    config=read(W/'platform-proxy-services.private.json')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==read(W/'v230-comparison/service-identity.json')['sha256']
    limits=read(ROOT/'source-freeze.json')['protocol']['limits']
    return base.ExactGenerationCache(config,ROOT,max_chat_calls=limits['chat_calls'],max_rerank_calls=limits['rerank_calls'],max_chat_tokens=limits['conservative_chat_byte_token_reservation'])

def judge(client,c,baseline,answer):
    data=[c['question'],c['golden'],baseline,answer]
    key=hashlib.sha256(json.dumps(data,ensure_ascii=False).encode()).hexdigest()
    result=base.pair(ROOT,client,'content-'+key,*data)
    return {'system':'v243_vs_v239','id':c['id'],**result}

def replay():
    assert not (ROOT/'v243-development-summary.json').exists()
    inspect=ROOT/'inspect-v243';inspect.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'packages/v243.zip') as z:z.extractall(inspect)
    with zipfile.ZipFile(ROOT/'packages/v242.zip') as z:tree=ast.parse(z.read('question_brief.py'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolved_paragraphs')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-v242-renderer>','exec'),namespace)
    old_resolve=namespace['resolved_paragraphs'];sys.path.insert(0,str(inspect))
    from question_brief import resolved_paragraphs
    from focused_context import render_paragraphs,date_note,search_queries
    from citation_labels import finish_answer
    from corpus import ingest
    docs={d.id:d for d in ingest(ROOT)[0]}
    cases=read(ROOT/'development.json');baseline=read(ROOT/'baseline-development.json')
    previous=read(ROOT/'development-answers.json')['v242_development']
    old_rows={r['id']:r for r in read(ROOT/'development-judgments.json')}
    answers={};proof=[];rows=[];jobs=[];start=time.monotonic()
    for c in cases:
        captured=read(ROOT/'generated/v242_development/captured'/(c['id']+'.json'))
        final=None;payload=None
        for call in reversed(captured['calls']):
            response=call.get('response',{})
            if response.get('paragraphs'):final=response;payload=call['payload'];break
            if call['schema']=='DraftCheck' and 'keep_prepared' in response:
                final=call['payload']['previous_draft'] if response['keep_prepared'] else response['correction']
                payload=call['payload'];break
        assert final and payload
        draft=SimpleNamespace(paragraphs=[SimpleNamespace(model_dump=lambda p=p:p) for p in final['paragraphs']])
        catalog=payload.get('citation_catalog',[]);evidence=payload['evidence']
        before=old_resolve(draft,catalog);after=resolved_paragraphs(draft,catalog)
        assert [p['citations'] for p in before]==[p['citations'] for p in after]
        def finish(paragraphs):
            text,refs=render_paragraphs(paragraphs,evidence)
            return finish_answer(date_note(c['question'],text,'2026-11-15',bool(refs)),search_queries(c['question'])[0],refs,evidence,docs)
        old=finish(before);new=finish(after)
        assert old==previous[c['id']]==captured['result']['answer'],c['id']
        answers[c['id']]=new;changed=new!=old
        proof.append({'id':c['id'],'exact_original_output_match':True,'changed':changed,
            'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),
            'references_unchanged':True})
        if changed:jobs.append(c)
        else:rows.append({**old_rows[c['id']],'system':'v243_vs_v239','reused_unchanged_output':True})
    write(ROOT/'v243-development-answers.json',{'v239_development':baseline,'v243_development':answers})
    write(ROOT/'v243-development-replay.json',proof)
    client=api();client.phase='v243-development-rendering'
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:rows+=list(pool.map(lambda c:judge(client,c,baseline[c['id']],answers[c['id']]),jobs))
    finally:client.close()
    rows.sort(key=lambda r:r['id']);write(ROOT/'v243-development-judgments.json',rows)
    summary={'phase':'development','aggregate':base.aggregate(rows),'rows':len(cases),'changed_outputs':[c['id'] for c in jobs],
        'fresh_generation_calls':0,'reused_generation_rows':18,'seconds':time.monotonic()-start,
        'cumulative_usage':base.usage(),'scope':'Exact replay of frozen v242 responses with UI label cleanup. All unchanged scores, including failures/losses, retained. Runtime timing remains the v242 full run.'}
    write(ROOT/'v243-development-summary.json',summary);print(json.dumps(summary),flush=True)

def holdout(plan):
    selection=read(ROOT/'selection.json')
    assert selection['runtime_freeze_sha256']==sha(ROOT/'runtime-freeze.json')
    assert selection['v243_evaluation_freeze_sha256']==sha(ROOT/'v243-evaluation-freeze.json')
    assert not (ROOT/'v243-holdout-summary.json').exists()
    cases=read(ROOT/'holdout.json');baseline=read(ROOT/'baseline-holdout.json')
    client=api();client.phase='v243_holdout';server=thread=cache=None;start=time.monotonic()
    try:
        cache=base.build_embeddings(ROOT,W/'bge-benchmark-weights',W/'official-warm-seed',None,plan)
        server=base.make_gateway(client,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        runtime=base.execute('v243_holdout',cases,plan,f'http://127.0.0.1:{server.server_port}/v1')
        answers=base.read_answers('v243_holdout');covers=base.coverage('v243_holdout',cases)
        write(ROOT/'v243-holdout-answers.json',{'v239_holdout':baseline,'v243_holdout':answers})
        write(ROOT/'v243-holdout-runtime.json',runtime);write(ROOT/'v243-holdout-coverage.json',covers)
        print(json.dumps({'runtime':'v243_holdout','seconds':runtime['seconds'],'counts':runtime['runtime_report'].get('counts'),'usage':base.usage()}),flush=True)
        if client.blocked or not runtime['complete'] or runtime['exit_code'] or runtime['runtime_report'].get('error_counts'):return 2
        client.phase='v243-holdout-judging'
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(lambda c:judge(client,c,baseline[c['id']],answers[c['id']]),cases))
        write(ROOT/'v243-holdout-judgments.json',rows)
        summary={'phase':'holdout','aggregate':base.aggregate(rows),'runtime':runtime,'seconds':time.monotonic()-start,
            'cumulative_usage':base.usage(),'qwen_cache_hits_phase':client.cache_hits,'generation_cache_hits_phase':client.gen_hits,
            'embedding_stats':cache.stats(),'scope':'Fresh full frozen v243 execution on 12 held-out questions; reused exact Qwen/BGE requests; immutable v239 baseline; same two-order judge.'}
        write(ROOT/'v243-holdout-summary.json',summary);print(json.dumps(summary),flush=True)
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        client.close()
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','holdout'],required=True);a=p.parse_args()
    plan=frozen()
    if a.phase=='replay':replay()
    else:raise SystemExit(holdout(plan))
