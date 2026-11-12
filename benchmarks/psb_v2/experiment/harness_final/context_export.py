"""Reconstruct actual complete source records from frozen code and trace IDs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def load_contexts(directory):
    directory=Path(directory)
    code="""import sys,json
sys.path.insert(0,sys.argv[1])
from legal_core import LegalAssistant,ModelProxy
proxy=ModelProxy();proxy.rerank_model=''
engine=LegalAssistant(sys.argv[2],proxy,use_embeddings=False)
print(json.dumps(engine.source_records,ensure_ascii=False))
"""
    env={key:value for key,value in os.environ.items() if key in ('PATH','HOME','LANG','LC_ALL')}
    records=json.loads(subprocess.check_output([sys.executable,'-c',code,str(directory),
        str(directory/'data/kb.zip')],cwd=directory,env=env,text=True,timeout=60))
    rows=[json.loads(line) for line in (directory/'submission.metrics.jsonl').read_text().splitlines()]
    contexts={}
    for row in rows:
        if 'id' not in row:continue
        events=[event for event in row.get('events',[]) if event.get('stage')=='context']
        if not events:
            contexts[row['id']]={'available':False,'sources':[]}
            continue
        # evidence-coverage may repack; the final context event is authoritative.
        event=events[-1];selected=[records[key] for key in event['source_ids']]
        characters=2+sum(len(json.dumps(record,ensure_ascii=False)) for record in selected)+max(0,len(selected)-1)*2
        if characters!=event['characters']:
            raise ValueError('Reconstructed context differs from recorded size: '+row['id'])
        contexts[row['id']]={'available':True,'sources':selected,'recorded_characters':characters}
    encoded=json.dumps(contexts,ensure_ascii=False,sort_keys=True).encode()
    return contexts,hashlib.sha256(encoded).hexdigest()


if __name__=='__main__':
    contexts,digest=load_contexts(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps({'sha256':digest,'contexts':contexts},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'questions':len(contexts),'sha256':digest}))
