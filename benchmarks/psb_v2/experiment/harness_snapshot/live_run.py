"""Keep local BGE service and actual submission children in one network namespace."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
from zipfile import ZipFile

import httpx
import numpy as np

ROOT=Path(__file__).resolve().parent


def merge_interrupted_vector_run(output, interrupted, remainder):
    """Recover completed stateless baseline answers, never claim one batch's timing."""
    old=json.loads((interrupted/'run.meta.json').read_text())
    new=json.loads((remainder/'run.meta.json').read_text())
    for key in ('version_id','commit','package_sha256','dataset_sha256','corpus_sha256','service_profile_sha256'):
        if old[key]!=new[key]:raise ValueError('Cannot merge different baseline configurations')
    if old['version_id']!='vector14' or new['exit_code']!=0 or not new['service_check_passed']:
        raise ValueError('Only a successfully resumed stateless baseline can be merged')
    old_metrics=[json.loads(s) for s in (interrupted/'submission.metrics.jsonl').read_text().splitlines()]
    new_metrics=[json.loads(s) for s in (remainder/'submission.metrics.jsonl').read_text().splitlines()]
    completed={row['id'] for row in old_metrics if 'id' in row}
    with (interrupted/'submission.csv').open() as f:answers={r['id']:r['answer'] for r in csv.DictReader(f) if r['id'] in completed}
    with (remainder/'submission.csv').open() as f:
        for row in csv.DictReader(f):
            if row['id'] in answers:raise ValueError('Repeated resumed question')
            answers[row['id']]=row['answer']
    if set(answers)!=set(old['ids']):raise ValueError('Incomplete merged answers')
    with (output/'submission.csv').open('w',newline='') as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL,lineterminator='\n');writer.writerow(['id','answer'])
        writer.writerows((key,answers[key]) for key in old['ids'])
    rows=[r for r in old_metrics+new_metrics if 'id' in r]
    (output/'submission.metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old_metrics+new_metrics))
    calls=[call for row in rows for call in row.get('api_calls',[])]
    old.update(exit_code=0,elapsed_seconds=None,interrupted_and_resumed=True,
        completion_mode='Merged completed questions from two actual solution.py executions',
        original_process_exit_code=None,resume_process_exit_code=new['exit_code'],
        single_batch_timing_valid=False,
        resume_segments=[str(interrupted.relative_to(output.parent)),str(remainder.relative_to(output.parent))],
        answers_sha256=hashlib.sha256((output/'submission.csv').read_bytes()).hexdigest(),
        service_check_passed=True,
        service_observations={'dense_index_ready':True,'rerank_api_calls':sum(c.get('endpoint')=='/rerank' for c in calls),
            'rerank_api_successes':sum(c.get('endpoint')=='/rerank' and c.get('http_status')==200 for c in calls),
            'technical_errors':sum(r.get('status')=='technical_error' for r in rows),'answer_rows':len(rows)})
    (output/'run.meta.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--services',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--dependency-path',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ids');p.add_argument('--split',default='development')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHONPATH':args.dependency_path}
    env['NO_PROXY']=env.get('NO_PROXY','')+',127.0.0.1,localhost'
    log=(args.output/'embedding-server.txt').open('w')
    server=subprocess.Popen([sys.executable,str(ROOT/'live_embedding_server.py'),'--weights',str(args.weights),
        '--seed-vectors',str(args.seed/'bge_embeddings.npz'),'--seed-manifest',str(args.seed/'embedding_cache.json'),
        '--log',str(args.output/'embedding.calls.jsonl'),'--fresh'],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        with httpx.Client(timeout=120,trust_env=False) as client:
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError('BGE process exited; see embedding-server.txt')
                try:
                    response=client.post('http://127.0.0.1:8769/v1/embeddings',json={'model':'BAAI/bge-m3','input':['Проверка модели.']})
                    response.raise_for_status();break
                except httpx.ConnectError:time.sleep(.5)
            else:raise RuntimeError('BGE startup timed out')
            print(json.dumps({'stage':'bge_ready'}),flush=True)
            with tempfile.TemporaryDirectory() as d:
                texts=[]
                for version in json.loads((args.frozen/'manifest.json').read_text())['versions']:
                    code=Path(d)/version['id'];code.mkdir()
                    with ZipFile(args.frozen/version['package']) as z:z.extractall(code)
                    script="import sys,json;sys.path.insert(0,sys.argv[1]);from legal_core import LegalAssistant,ModelProxy,focused_search_question;from rag_safety import clean_data;p=ModelProxy();p.rerank_model='';e=LegalAssistant(sys.argv[2],p,use_embeddings=False);rows=[json.loads(s) for s in open(sys.argv[3])];print(json.dumps(e.search_texts+[focused_search_question(clean_data(r['question'])[0]) for r in rows if r['split']==sys.argv[4]]))"
                    texts.extend(json.loads(subprocess.check_output([sys.executable,'-c',script,str(code),str(ROOT/'knowledge-base.zip'),str(ROOT/'cases.jsonl'),args.split],text=True)))
                texts=list(dict.fromkeys(texts))
                for i in range(0,len(texts),16):
                    r=client.post('http://127.0.0.1:8769/v1/embeddings',json={'model':'BAAI/bge-m3','input':texts[i:i+16]});r.raise_for_status()
                old=json.loads((args.seed/'embedding_cache.json').read_text())
                new=client.post('http://127.0.0.1:8769/v1/embeddings',json={'model':'BAAI/bge-m3','input':[texts[0]]}).json()['data'][0]['embedding']
                with np.load(args.seed/'bge_embeddings.npz',allow_pickle=False) as z:
                    original=z['vectors'][old['text_sha256_to_row'][hashlib.sha256(texts[0].encode()).hexdigest()]]
                delta=float(np.max(np.abs(original-np.asarray(new))))
                verification={'texts':len(texts),'max_absolute_old_cache_difference':delta,
                              'old_cache_cosine':float(np.dot(original,np.asarray(new))),
                              'old_vectors_used':False,'all_inputs_reencoded':True,'model':old['model'],'revision':old['revision'],
                              'timing':'local CPU service with warm text-hash cache; not organizer embedding latency'}
                (args.output/'embedding_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
                print(json.dumps({'stage':'bge_verified',**verification}),flush=True)
        child_env=dict(os.environ);child_env['NO_PROXY']=env['NO_PROXY']
        for version in ('vector14','conditional35','coverage2257','review29'):
            output=args.output/version
            interrupted=None
            run_output=output
            run_ids=args.ids
            if (output/'run.meta.json').exists():
                meta=json.loads((output/'run.meta.json').read_text())
                if 'exit_code' in meta:
                    print(json.dumps({'stage':'existing_run','version':version}),flush=True);continue
                if version!='vector14':raise ValueError('Incomplete stateful run exists; use a new destination')
                interrupted=args.output/'segments'/'vector14-interrupted'
                if not interrupted.exists():shutil.copytree(output,interrupted)
                rows=[json.loads(s) for s in (interrupted/'submission.metrics.jsonl').read_text().splitlines()]
                completed={r['id'] for r in rows if 'id' in r}
                run_ids=','.join(key for key in meta['ids'] if key not in completed)
                if not run_ids:raise ValueError('No missing questions; inspect interrupted completion separately')
                run_output=args.output/'segments'/'vector14-remainder'
            command=[sys.executable,str(ROOT/'run_matrix.py'),'run','--frozen',str(args.frozen),'--version',version,
                     '--services',str(args.services),'--split',args.split,'--output',str(run_output)]
            if run_ids:command.extend(['--ids',run_ids])
            with (args.output/(version+'.launcher.txt')).open('w') as stream:
                proc=subprocess.Popen(command,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
                print(json.dumps({'stage':'generation_start','version':version}),flush=True)
                previous=-1;last=0
                while proc.poll() is None:
                    time.sleep(.5)
                    now=time.monotonic()
                    if now-last<10:continue
                    last=now;metrics=run_output/'submission.metrics.jsonl'
                    count=0
                    if metrics.exists():
                        for line in metrics.read_text().splitlines():
                            try:count+=int('id' in json.loads(line))
                            except json.JSONDecodeError:pass
                    if count!=previous:
                        print(json.dumps({'stage':'generation_progress','version':version,'finished':count}),flush=True);previous=count
                if proc.returncode:raise RuntimeError('Submission wrapper failed; see launcher log')
            if interrupted:merge_interrupted_vector_run(output,interrupted,run_output)
            meta=json.loads((output/'run.meta.json').read_text())
            print(json.dumps({'stage':'generation_complete','version':version,'elapsed_seconds':meta['elapsed_seconds'],
                              'service_check_passed':meta['service_check_passed'],'observations':meta['service_observations']}),flush=True)
            if not meta['service_check_passed']:raise RuntimeError('Reranker/dense execution check failed')
    finally:
        server.terminate()
        try:server.wait(timeout=5)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()


if __name__=='__main__':main()
