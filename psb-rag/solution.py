"""PSB flat-archive entry point. Each CSV row is an independent request."""
import argparse, csv, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from settings import Settings
from pipeline import Pipeline,TECHNICAL

def run(input_path=None,output_path='submission.csv',pipeline=None):
    settings=pipeline.s if pipeline else Settings()
    input_path=Path(input_path or settings.data_dir/'questions.csv')
    with input_path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','question'] and not {'id','question'}.issubset(reader.fieldnames or []): raise ValueError('Input must contain id,question')
        rows=list(reader)
    seen=set(); unique=[]
    for row in rows:
        if row['id'] not in seen: unique.append(row); seen.add(row['id'])
    results={r['id']:TECHNICAL for r in unique}; statuses={}
    started=time.monotonic()
    def save():
        path=Path(output_path); tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['id','answer'],quoting=csv.QUOTE_ALL); writer.writeheader()
            writer.writerows({'id':r['id'],'answer':results[r['id']]} for r in unique)
        tmp.replace(path)
    save()  # Valid full CSV exists even if model/bootstrap fails later.
    try: core=pipeline or Pipeline(settings); core.initialize()
    except Exception:
        return {'rows':len(unique),'status':'initialization_error','seconds':time.monotonic()-started}
    def answer(row):
        try: return core.ask(row['question']) # No session ID in batch mode.
        except Exception: return {'answer':TECHNICAL,'status':'technical_error'}
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
                statuses[row['id']]=result['status']; save(); schedule()
    finally:
        for future in pending: future.cancel()
        executor.shutdown(wait=False,cancel_futures=True); save()
    report={'rows':len(unique),'statuses':statuses,'seconds':round(time.monotonic()-started,4)}
    Path(output_path).with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input'); parser.add_argument('--output',default='submission.csv')
    args=parser.parse_args(); print(json.dumps(run(args.input,args.output),ensure_ascii=True))
