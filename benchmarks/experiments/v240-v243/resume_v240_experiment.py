"""Continue HTTP-402 interruption without rerunning successful answers.

Original frozen harness/packages and interrupted artifacts remain unchanged.
Only missing reviews and seven unanswered cases are run. No answer selection.
"""
import argparse,copy,csv,json,shutil,time
from collections import Counter
from pathlib import Path
import run_v240_experiment as base

ROOT=base.ROOT
ORIGINAL='v239_development'
RECOVERED='v239_development_recovered'
REMAINDER='v239_development_remainder'
original_execute=base.execute
original_read_answers=base.read_answers
original_coverage=base.coverage
alias_ready=False

def snapshot():
    target=ROOT/'interrupted-first-run'
    if target.exists():return
    target.mkdir()
    files=['development-runtime.json','development-answers.json','development-coverage.json','calls.jsonl']
    hashes={}
    for name in files:
        source=ROOT/name
        if source.exists():shutil.copyfile(source,target/name);hashes[name]=base.sha(source)
    base.write(target/'manifest.json',{'files':hashes,'original_directory':ORIGINAL,
        'original_execution_sha256':base.sha(ROOT/'generated'/ORIGINAL/'execution.json')})

def review_only(ident,url):
    from settings import Settings
    from pipeline import Pipeline,Answer
    from focused_context import render_paragraphs,date_note,search_queries
    from citation_labels import finish_answer
    old_file=ROOT/'generated'/ORIGINAL/'captured'/(ident+'.json')
    output=ROOT/'review-continuations';output.mkdir(exist_ok=True)
    target=output/(ident+'.json')
    if target.exists():
        value=base.read(target)
        assert value['recovery']['original_capture_sha256']==base.sha(old_file)
        assert value['result']['trace']['review']=='applied'
        return value
    old=base.read(old_file);calls=old['calls']
    assert len(calls)==2 and calls[0].get('response') and 'response' not in calls[1]
    failed=calls[1];assert failed['schema']=='Answer' and 'previous_draft' in failed['payload']
    assert failed['payload']['previous_draft']==calls[0]['response']
    assert old['result']['trace']['review']=='failed_kept_draft'
    settings=Settings(base_url=url,api_key='local-runtime',model=base.DEEPSEEK,
        emb_url=url,emb_key='local-runtime',rerank_url=url,rerank_key='local-runtime',rerank_model=base.QWEN,
        data_dir=ROOT/'generated'/ORIGINAL/'data',cache_dir=output/'cache',as_of='2026-11-15',
        max_tokens=8192,temperature=0,deadline_seconds=1100)
    core=Pipeline(settings);start=time.monotonic()
    try:
        revised=core.api.structured(failed['instruction'],failed['payload'],Answer)
        evidence=failed['payload']['evidence']
        answer,refs=render_paragraphs([p.model_dump() for p in revised.paragraphs],evidence)
        assert revised.status not in ('answered','partial') or refs,'Reviewed answer lacks citations'
        answer=date_note(old['question'],answer,settings.as_of,bool(refs))
        answer=finish_answer(answer,search_queries(old['question'])[0],refs,evidence,core.docs)
        value=copy.deepcopy(old)
        record={k:copy.deepcopy(failed[k]) for k in ('schema','instruction','payload')}
        record.update(response=revised.model_dump(),seconds=time.monotonic()-start,resumed_missing_review=True)
        value['calls'].append(record)
        value['result'].update(answer=answer,status=revised.status)
        value['result']['trace']['review']='applied'
        value['recovery']={'original_capture_sha256':base.sha(old_file),'reused_initial_draft':True,
            'resumed_stage':'review','seconds':record['seconds'],'original_failure_retained_in_calls':True}
        value['seconds']+=record['seconds'];base.write(target,value);return value
    finally:core.close()

def execute(name,cases,plan,url,oracle=False):
    global alias_ready
    if name!=ORIGINAL:return original_execute(name,cases,plan,url,oracle)
    directory=ROOT/'generated'/RECOVERED
    if (directory/'execution.json').exists():
        meta=base.read(directory/'execution.json')
        assert meta['complete'] and meta['answers_sha256']==base.sha(directory/'submission.csv')
        alias_ready=True;return meta
    source=ROOT/'generated'/ORIGINAL
    original_meta=base.read(source/'execution.json')
    assert set(original_meta['runtime_report']['error_counts'])=={'http_402'}
    completed={};review_ids=[];missing=[]
    for c in cases:
        old=base.read(source/'captured'/(c['id']+'.json'))
        if old.get('result'):
            if old['result']['trace']['review']=='applied':completed[c['id']]=old
            else:
                assert old['result']['trace']['review']=='failed_kept_draft'
                review_ids.append(c['id'])
        else:missing.append(c)
    assert len(completed)==9 and len(review_ids)==2 and len(missing)==7
    recovered={ident:review_only(ident,url) for ident in review_ids}
    tail=original_execute(REMAINDER,missing,plan,url,False)
    if not tail['complete'] or tail['exit_code'] or tail['runtime_report'].get('error_counts'):
        raise RuntimeError('Remainder failed; artifacts retained, no repeated successful responses')
    for c in missing:
        recovered[c['id']]=base.read(ROOT/'generated'/REMAINDER/'captured'/(c['id']+'.json'))
    values={**completed,**recovered};assert set(values)=={c['id'] for c in cases}
    directory.mkdir(exist_ok=True);capture=directory/'captured';capture.mkdir(exist_ok=True)
    for ident,value in values.items():
        if ident in completed:shutil.copyfile(source/'captured'/(ident+'.json'),capture/(ident+'.json'))
        else:base.write(capture/(ident+'.json'),value)
    with (directory/'submission.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','answer'])
        writer.writerows((c['id'],values[c['id']]['result']['answer']) for c in cases)
    original_answers=original_read_answers(ORIGINAL);new_answers=original_read_answers(RECOVERED)
    assert all(original_answers[k]==new_answers[k] for k in completed)
    statuses={c['id']:values[c['id']]['result']['status'] for c in cases}
    recovery={'strategy':'Nine completed answers unchanged; two exact missing reviews; seven unanswered cases only.',
        'completed_reused':list(completed),'review_only':review_ids,'remainder':[c['id'] for c in missing],
        'original_execution_sha256':base.sha(source/'execution.json'),
        'remainder_execution_sha256':base.sha(ROOT/'generated'/REMAINDER/'execution.json'),
        'timing_note':'Sum of interrupted run, review continuations and remainder; not a fresh uninterrupted benchmark.'}
    seconds=original_meta['seconds']+tail['seconds']+sum(values[k]['recovery']['seconds'] for k in review_ids)
    report={'rows':len(cases),'statuses':statuses,'counts':dict(Counter(statuses.values())),
        'error_counts':{},'seconds':round(seconds,3),'recovery':recovery}
    meta={'system':ORIGINAL,'seconds':round(seconds,3),'exit_code':0,'complete':True,'outer_timeout':False,
        'rows':len(cases),'answers_sha256':base.sha(directory/'submission.csv'),
        'package_sha256':original_meta['package_sha256'],'mode':'resumed_original_runtime',
        'runtime_report':report,'recovery':recovery}
    base.write(directory/'submission.report.json',report);base.write(directory/'execution.json',meta)
    alias_ready=True;return meta

def read_answers(name):
    return original_read_answers(RECOVERED if alias_ready and name==ORIGINAL else name)

def coverage(name,cases):
    return original_coverage(RECOVERED if alias_ready and name==ORIGINAL else name,cases)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','holdout'],required=True)
    args=parser.parse_args();snapshot()
    proof={'script_sha256':base.sha(__file__),'runtime_freeze_sha256':base.sha(ROOT/'runtime-freeze.json'),
        'scope':'Transport recovery only; frozen packages, questions, judge, successful answers unchanged.'}
    frozen=ROOT/'recovery-freeze.json'
    if frozen.exists():assert base.read(frozen)==proof
    else:base.write(frozen,proof)
    base.execute=execute;base.read_answers=read_answers;base.coverage=coverage
    raise SystemExit(base.main(args))
