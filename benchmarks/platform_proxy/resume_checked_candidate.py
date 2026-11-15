"""Explicit recovery of failed rows; never rerun already successful judgments.

Run only after the DeepSeek access problem is resolved. Original failed run and
its ledger remain unchanged; this new run replaces only failed/degraded rows.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import threading
import time
from prepare import ROOT,sha,write
from run import check,judge_pair,summarize
from services import Services
from stress_runtime import build_embeddings,make_gateway,execute

class CircuitServices(Services):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.blocked={};self.block_lock=threading.Lock()
    def post(self,role,payload):
        group='rerank' if role=='RERANKER' else 'chat'
        with self.block_lock:
            if 'chat' in self.blocked:raise RuntimeError(self.blocked['chat'])
            if group in self.blocked:raise RuntimeError(self.blocked[group])
        try:return super().post(role,payload)
        except RuntimeError as exc:
            message=str(exc)
            if any(message.startswith('API HTTP '+str(code)) for code in (401,402,403)):
                with self.block_lock:self.blocked[group]=message
            raise

def load_answers(path):
    with path.open(encoding='utf8',newline='') as f:return {r['id']:r['answer'] for r in csv.DictReader(f)}

def main(a):
    config=json.loads(a.services.read_text(encoding='utf8'));parent=check(a.parent,config)
    cases=json.loads((a.parent/'cases.json').read_text(encoding='utf8'))
    original_system=parent['systems'][0];name=original_system+'_resumed'
    old_runtime=json.loads((a.parent/'runtime.json').read_text(encoding='utf8'))[original_system]
    old_dir=a.parent/'generated'/original_system
    ids={k for k,v in old_runtime['runtime_report']['statuses'].items() if v in ('technical_error','deadline_inflight','deadline_unprocessed')}
    for line in (old_dir/'cache/diagnostics.jsonl').read_text(encoding='utf8').splitlines():
        row=json.loads(line)
        if row.get('event')=='review_degraded_kept_draft':ids.add(row['request_id'])
    root=a.output
    if not root.exists():
        root.mkdir(parents=True);(root/'packages').mkdir()
        for n in ('cases.json','baseline.json','knowledge-base.zip','historical-platform.json','service-identity.json','embedding-cache.sqlite3'):
            shutil.copyfile(a.parent/n,root/n)
        shutil.copyfile(a.parent/'packages'/f'{original_system}.zip',root/'packages'/f'{name}.zip')
        write(root/'previous-answers.json',load_answers(old_dir/'submission.csv'))
        plan={'version':'focused-explicit-recovery','systems':[name],'cases':parent['cases'],
              'scope':'Recovery after HTTP 402 on unchanged development cases and runtime.',
              'input_hashes':{n:sha(root/n) for n in ('cases.json','baseline.json','knowledge-base.zip','previous-answers.json')},
              'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},
              'embedding_code_hashes':parent['embedding_code_hashes'],'recovery_ids':sorted(ids),
              'packages':{name:parent['packages'][original_system]},'parent_freeze_sha256':sha(a.parent/'freeze.json'),
              'differences_from_psb':parent['differences_from_psb'],
              'api_limits':{'chat_calls':80,'rerank_calls':15,'conservative_chat_token_ceiling':2500000}}
        write(root/'freeze.json',plan)
    plan=check(root,config)
    api=CircuitServices(config,root,max_chat_calls=80,max_rerank_calls=15,max_chat_tokens=2500000)
    cache=server=thread=None;start=time.monotonic()
    try:
        cache=build_embeddings(root,a.weights,a.seed,None,plan)
        server=make_gateway(api,cache);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        pending=[c for c in cases if c['id'] in plan['recovery_ids']]
        print(json.dumps({'recovering_ids':plan['recovery_ids'],'unchanged_zip':plan['packages'][name]['sha256']}),flush=True)
        meta=execute(root,name,pending,plan,f'http://127.0.0.1:{server.server_port}/v1')
        write(root/'runtime.json',{name:meta})
        if not meta['complete'] or meta.get('runtime_report',{}).get('error_counts') or api.blocked:
            print('Recovery incomplete; judge calls skipped.',flush=True);return 2
        answers=json.loads((root/'previous-answers.json').read_text(encoding='utf8'))
        answers.update(load_answers(root/'generated'/name/'submission.csv'))
        write(root/'all-answers.json',{name:answers})
        base=json.loads((root/'baseline.json').read_text(encoding='utf8'))
        def judge(c):
            result=judge_pair(root,api,name+'-'+c['id'],c['question'],c['golden'],base[c['id']]['value']['answer'],answers[c['id']])
            print(json.dumps({'judged':c['id'],'status':result['status'],'score':result.get('score')}),flush=True)
            return {'system':name,'id':c['id'],'family_id':c['family_id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(judge,cases))
        write(root/'judgments.json',rows)
        report=summarize(root,plan,rows,time.monotonic()-start)
        report['recovered_ids']=plan['recovery_ids'];report['runtime']={name:meta}
        report['timing_note']='Recovery duration, not a fresh full-batch latency measurement.'
        write(root/'summary.json',report)
        print(json.dumps({'systems':report['systems'],'seconds':report['seconds']}),flush=True)
        return 0
    finally:
        if server:server.shutdown();server.server_close();thread.join()
        if cache:cache.close()
        api.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('parent','output','services','weights','seed'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
