"""Local OpenAI-shaped BGE-M3 endpoint. No model substitution or product edits.

Uses hash-verified old vectors when exact input text matches; otherwise encodes
with pinned official BGE-M3 weights. Timing is NOT a platform API reproduction.
"""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--seed-vectors',type=Path,required=True)
    p.add_argument('--seed-manifest',type=Path,required=True)
    p.add_argument('--port',type=int,default=8769)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--fresh',action='store_true')
    args=p.parse_args()
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    meta=json.loads(args.seed_manifest.read_text())
    if hashlib.sha256(args.seed_vectors.read_bytes()).hexdigest()!=meta['cache_sha256']:
        raise ValueError('Corrupt embedding cache')
    with np.load(args.seed_vectors,allow_pickle=False) as data:matrix=data['vectors']
    if matrix.shape[1]!=1024 or not np.isfinite(matrix).all():raise ValueError('Invalid embeddings')
    cache={key:matrix[index].tolist() for key,index in meta['text_sha256_to_row'].items()}
    if args.fresh:cache={}
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(args.weights,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(args.weights,local_files_only=True,trust_remote_code=False,
                                     torch_dtype=torch.float32,attn_implementation='sdpa').eval()
    lock=Lock()
    args.log.parent.mkdir(parents=True,exist_ok=True)
    log=args.log.open('w',encoding='utf-8',buffering=1)

    def embed(texts):
        start=time.perf_counter();keys=[hashlib.sha256(t.encode()).hexdigest() for t in texts]
        with lock:
            pending=list(dict.fromkeys(key for key in keys if key not in cache))
            for offset in range(0,len(pending),8):
                batch=pending[offset:offset+8]
                strings=[texts[keys.index(key)] for key in batch]
                tokens=tokenizer(strings,padding=True,truncation=False,return_tensors='pt')
                if tokens['input_ids'].shape[1]>8192:raise ValueError('Input exceeds BGE context')
                with torch.inference_mode():
                    encoded=model(**tokens).last_hidden_state[:,0]
                    encoded=torch.nn.functional.normalize(encoded,p=2,dim=1).cpu().numpy()
                for key,vector in zip(batch,encoded):cache[key]=vector.tolist()
            log.write(json.dumps({'texts':len(texts),'new_texts':len(pending),
                                  'elapsed_ms':round((time.perf_counter()-start)*1000,3),
                                  'input_hashes':keys})+'\n')
            return [cache[key] for key in keys]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                if self.path!='/v1/embeddings':raise ValueError('Unknown endpoint')
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if data.get('model','').lower()!='baai/bge-m3':raise ValueError('Wrong model')
                texts=data['input'];texts=[texts] if isinstance(texts,str) else texts
                if not isinstance(texts,list) or not texts or not all(isinstance(t,str) for t in texts):raise ValueError('Invalid input')
                vectors=embed(texts)
                response={'model':'BAAI/bge-m3','data':[{'index':i,'embedding':v} for i,v in enumerate(vectors)]}
                status=200
            except Exception as exc:
                response={'error':{'type':type(exc).__name__}};status=400
            body=json.dumps(response).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(json.dumps({'ready':True,'model':'BAAI/bge-m3','revision':meta['revision'],
                      'cached_texts':len(cache),'load_seconds':round(time.perf_counter()-start,2),
                      'port':args.port,'torch':torch.__version__}),flush=True)
    try:server.serve_forever()
    finally:log.close();server.server_close()


if __name__=='__main__':main()
