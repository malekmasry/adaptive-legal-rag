"""PSB flat-archive entry point. Each CSV row is an independent request."""
import argparse, csv, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter
from settings import Settings
from pipeline import Pipeline,TECHNICAL
from diagnostics import emit,error_info,request_scope,stage_scope

def run(input_path=None,output_path='submission.csv',pipeline=None):
    settings=pipeline.s if pipeline else Settings()
    emit(settings.cache_dir,'run_configuration',max_tokens=settings.max_tokens,
         max_tokens_from_environment=bool(os.environ.get('MAX_TOKENS','').strip()),
         request_timeout=settings.request_timeout,run_budget_seconds=settings.deadline_seconds,
         workers=settings.workers,top_k=settings.top_k,rerank_top_n=settings.rerank_n,
         context_tokens=settings.context_tokens,temperature=settings.temperature,
         generator=settings.model,embedder=settings.emb_model,reranker=settings.rerank_model)
    input_path=Path(input_path or settings.data_dir/'questions.csv')
    with input_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','question'] and not {'id','question'}.issubset(reader.fieldnames or []): raise ValueError('Input must contain id,question')
        rows=list(reader)
    seen=set(); unique=[]
    for row in rows:
        if row['id'] not in seen: unique.append(row); seen.add(row['id'])
    results={r['id']:TECHNICAL for r in unique}; statuses={}; errors={}
    started=time.monotonic()
    def save():
        path=Path(output_path); tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','answer'],quoting=csv.QUOTE_ALL); writer.writeheader()
            writer.writerows({'id':r['id'],'answer':results[r['id']]} for r in unique)
        tmp.replace(path)
    save()  # Valid full CSV exists even if model/bootstrap fails later.
    try:
        core=pipeline or Pipeline(settings)
        if hasattr(core.api,'deadline'):core.api.deadline=started+settings.deadline_seconds
        core.initialize()
    except Exception as exc:
        emit(settings.cache_dir,'initialization_failed',**error_info(exc))
        report={'rows':len(unique),'status':'initialization_error','seconds':time.monotonic()-started,
                'error':error_info(exc),'statuses':{r['id']:'initialization_error' for r in unique}}
        Path(output_path).with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        return report
    def answer(row):
        with request_scope(row['id']),stage_scope('answer'):
            emit(settings.cache_dir,'question_started')
            try: return core.ask(row['question']) # No session ID in batch mode.
            except Exception as exc:
                emit(settings.cache_dir,'question_failed',**error_info(exc))
                return {'answer':TECHNICAL,'status':'technical_error',
                        'trace':{'failures':[{'stage':'unhandled_question',**error_info(exc)}]}}
    executor=ThreadPoolExecutor(max_workers=settings.workers)
    iterator=iter(unique); pending={}
    def schedule():
        try:
            row=next(iterator); pending[executor.submit(answer,row)]=row
        except StopIteration: pass
    for _ in range(settings.workers): schedule()
    try:
        while pending and time.monotonic()-started<settings.deadline_seconds:
            done,_=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
            for future in done:
                row=pending.pop(future); result=future.result(); value=result['answer']
                if value.strip() and len(value)<=10000: results[row['id']]=value
                else:
                    result['status']='technical_error'
                    result.setdefault('trace',{}).setdefault('failures',[]).append(
                        {'stage':'csv_output','error_code':'invalid_final_answer'})
                failures=result.get('trace',{}).get('failures',[])
                if failures:errors[row['id']]=failures
                statuses[row['id']]=result['status']; save(); schedule()
    finally:
        for future in pending: future.cancel()
        executor.shutdown(wait=False,cancel_futures=True); save()
    for row in unique:
        if row['id'] not in statuses:
            statuses[row['id']]='deadline_unprocessed'
            errors[row['id']]=[{'stage':'scheduler','error_code':'run_deadline'}]
            with request_scope(row['id']):emit(settings.cache_dir,'question_not_completed',error_code='run_deadline')
    report={'rows':len(unique),'statuses':statuses,'counts':dict(Counter(statuses.values())),
            'seconds':round(time.monotonic()-started,4),'errors':errors,
            'error_counts':dict(Counter(e['error_code'] for values in errors.values() for e in values))}
    Path(output_path).with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    emit(settings.cache_dir,'run_completed',rows=len(unique),counts=report['counts'],seconds=report['seconds'],error_counts=report['error_counts'])
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input'); parser.add_argument('--output',default='submission.csv')
    args=parser.parse_args(); print(json.dumps(run(args.input,args.output),ensure_ascii=True))
