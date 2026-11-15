"""Run unchanged submission ZIPs behind an API role firewall and local BGE."""
import csv
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile
import numpy as np

from prepare import ROOT,sha,write
from services import DEEPSEEK,QWEN,BudgetExceeded


def build_embeddings(root,weights,seed,pilot,plan):
    engine=ROOT.parent/'v8_deepseek/engine'
    for name,digest in plan['embedding_code_hashes'].items():
        if sha(engine/name)!=digest:raise ValueError('Embedding implementation changed')
    sys.path.insert(0,str(engine))
    from embedding_cache import EmbeddingCache,validate_vectors
    from live_embedding_server import LazyBGE
    cache=EmbeddingCache(seed/'bge_embeddings.npz',seed/'embedding_cache.json',root/'embedding-cache.sqlite3',encoder=LazyBGE(weights))
    if pilot:
        manifest=json.loads((pilot/'embedding-cache.json').read_text(encoding='utf8'))
        if manifest['model']!='BAAI/bge-m3' or manifest['revision']!='5617a9f61b028005a4858fdac845db406aefb181':
            raise ValueError('Pilot embeddings differ')
        with np.load(pilot/'embedding-cache.npz',allow_pickle=False) as data:
            matrix=validate_vectors(data['vectors'],len(manifest['texts']))
        for text,vector in zip(manifest['texts'],matrix):
            key=hashlib.sha256(text.encode()).hexdigest()
            if key in cache.vectors and not np.allclose(cache.vectors[key],vector,atol=1e-5):
                raise ValueError('Conflicting genuine embedding cache')
            cache.vectors[key]=vector.tolist()
    return cache


def make_gateway(api,embeddings):
    semaphore=threading.Semaphore(3)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            status=200
            try:
                if self.headers.get('Authorization')!='Bearer local-runtime':raise PermissionError('Unknown local client')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<8_000_000:raise ValueError('Invalid payload size')
                data=json.loads(self.rfile.read(size))
                if self.path=='/v1/embeddings':
                    if data.get('model')!='BAAI/bge-m3':raise ValueError('Wrong local embedding model')
                    texts=data['input'];texts=[texts] if isinstance(texts,str) else texts
                    vectors,_=embeddings.embed(texts)
                    response={'model':'BAAI/bge-m3','data':[{'index':i,'embedding':v} for i,v in enumerate(vectors)]}
                elif self.path in ('/v1/chat/completions','/v1/rerank'):
                    role='GENERATOR' if self.path.endswith('chat/completions') else 'RERANKER'
                    with semaphore:response=api.post(role,data)
                else:raise ValueError('Unknown route')
            except PermissionError:response={'error':{'type':'forbidden'}};status=403
            except BudgetExceeded:response={'error':{'type':'budget_exhausted'}};status=402
            except ValueError:response={'error':{'type':'invalid_role_or_payload'}};status=400
            except Exception as exc:
                response={'error':{'type':type(exc).__name__}};status=502
                # Preserve upstream permanent failures so the unchanged runtime
                # does not turn one denial into three new upstream attempts.
                if str(exc).startswith('API HTTP '):
                    code=str(exc).rsplit(' ',1)[-1]
                    if code.isdigit():status=int(code)
            body=json.dumps(response).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
    return ThreadingHTTPServer(('127.0.0.1',0),Handler)


def kill_tree(process):
    if process.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    else:os.killpg(process.pid,signal.SIGKILL)


def execute(root,name,cases,plan,gateway_url):
    directory=root/'generated'/name
    archive=root/'packages'/(name+'.zip')
    if sha(archive)!=plan['packages'][name]['sha256']:raise ValueError('Frozen ZIP changed')
    if directory.exists():
        meta=json.loads((directory/'execution.json').read_text(encoding='utf8')) if (directory/'execution.json').exists() else {}
        if meta.get('complete') and meta.get('answers_sha256')==sha(directory/'submission.csv'):
            return meta
        raise ValueError('Incomplete runtime retained; refusing automatic paid rerun')
    directory.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        if any(Path(n).name!=n or n in ('','.','..') for n in z.namelist()):raise ValueError('Non-flat runtime')
        z.extractall(directory)
    data=directory/'data';data.mkdir()
    (data/'knowledge-base.zip').write_bytes((root/'knowledge-base.zip').read_bytes())
    with (data/'questions.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','question']);writer.writerows((c['id'],c['question']) for c in cases)
    env={k:v for k,v in os.environ.items() if k.upper() in ('PATH','SYSTEMROOT','WINDIR','TEMP','TMP','HOME','USERPROFILE')}
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',DATA_DIR=str(data),CACHE_DIR=str(directory/'cache'),
        OUTPUT_CSV=str(directory/'submission.csv'),LLM_MODEL=DEEPSEEK,LLM_BASE_URL=gateway_url,LLM_API_KEY='local-runtime',
        EMB_MODEL='BAAI/bge-m3',EMBEDDER_BASE_URL=gateway_url,EMBEDDER_API_KEY='local-runtime',
        RERANKER_MODEL=QWEN,RERANKER_BASE_URL=gateway_url,RERANKER_API_KEY='local-runtime',
        TOP_K=str(plan['packages'][name]['top_k']),MAX_WORKERS='3',MAX_TOKENS='8192',TEMPERATURE='0',
        RUN_BUDGET_SECONDS='1100',AS_OF='2026-11-14',CHUNKER='common',NO_PROXY='127.0.0.1,localhost')
    start=time.monotonic();timed_out=False;process=None
    try:
        with (directory/'stdout.txt').open('wb') as out,(directory/'stderr.txt').open('wb') as err:
            kwargs={'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {'start_new_session':True}
            process=subprocess.Popen([sys.executable,'-X','utf8','solution.py'],cwd=directory,env=env,stdout=out,stderr=err,**kwargs)
            try:code=process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                timed_out=True;kill_tree(process);code=process.wait(timeout=20)
    finally:
        if process is not None:kill_tree(process)
    file=directory/'submission.csv';answers={}
    if file.exists():
        with file.open(encoding='utf8',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['id'] in answers:raise ValueError('Duplicate output ID')
                answers[row['id']]=row['answer']
    complete=set(answers)=={c['id'] for c in cases}
    meta={'system':name,'seconds':round(time.monotonic()-start,3),'exit_code':code,'outer_timeout':timed_out,
        'complete':complete,'answers_sha256':sha(file) if file.exists() else None,
        'package_sha256':sha(archive),'rows':len(answers),'max_workers':3,'top_k':env['TOP_K'],
        'credentials_exposed_to_runtime':False,'runtime_changed':False}
    if (directory/'submission.report.json').exists():meta['runtime_report']=json.loads((directory/'submission.report.json').read_text(encoding='utf8'))
    write(directory/'execution.json',meta)
    return meta
