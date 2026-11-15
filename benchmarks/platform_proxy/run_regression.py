"""Generate additional unseen formulations, with no model judge or baseline cost."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from prepare import ROOT,sha,write
from run import check
from services import Services
from stress_runtime import build_embeddings,make_gateway,execute

def main(a):
    config=json.loads(a.services.read_text(encoding='utf8'))
    parent=check(a.parent,config);root=a.output
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for n in ('embedding-cache.sqlite3','knowledge-base.zip','service-identity.json'):
            shutil.copyfile(a.parent/n,root/n)
        shutil.copyfile(a.cases,root/'cases.json')
        shutil.copyfile(a.package,root/'packages/regression.zip')
        write(root/'freeze.json',{'version':'focused-regression10','embedding_code_hashes':parent['embedding_code_hashes'],
            'input_hashes':{n:sha(root/n) for n in ('knowledge-base.zip','cases.json')},
            'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},
            'packages':{'regression':{'sha256':sha(a.package),'top_k':40}},
            'evaluation':'Manual assessment of new formulations; no target answers enter runtime',
            'api_limits':{'chat_calls':20,'rerank_calls':15,'conservative_chat_token_ceiling':1800000}})
    plan=check(root,config)
    if sha(a.package)!=plan['packages']['regression']['sha256']:raise ValueError('Changed package')
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    api=Services(config,root,max_chat_calls=20,max_rerank_calls=15,max_chat_tokens=1800000)
    cache=server=thread=None;start=time.monotonic()
    try:
        cache=build_embeddings(root,a.weights,a.seed,None,plan)
        sys.path.insert(0,str(ROOT.parents[1]/'focused-rag'))
        from focused_context import search_queries
        queries=[]
        for c in cases:
            q,sub=search_queries(c['question']);queries.extend([q]+sub)
        warm=time.monotonic();cache.embed(list(dict.fromkeys(queries)))
        write(root/'embedding-warmup.json',{'seconds':time.monotonic()-warm,'stats':cache.stats()})
        print('Embeddings ready',flush=True)
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        meta=execute(root,'regression',cases,plan,f'http://127.0.0.1:{server.server_port}/v1')
        write(root/'runtime.json',meta)
        print(json.dumps({'complete':meta['complete'],'seconds':time.monotonic()-start,'runtime':meta}),flush=True)
        return 0 if meta['complete'] else 1
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('parent','output','cases','package','services','weights','seed'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
