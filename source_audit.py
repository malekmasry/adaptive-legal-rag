from prompt_config import PROMPTS
"""Model-assisted source instruction detection; never rewrites or deletes source data."""
import json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from pydantic import Field
from schemas import Strict
from model_api import ModelError

VERSION='source-instructions-v1'
PROMPT=PROMPTS['source_classifier']

class SourceClassification(Strict):
    chunk_id:str
    instruction_quotes:list[str]=Field(default_factory=list)
class SourceAudit(Strict):
    items:list[SourceClassification]

def fingerprint(chunks):
    entries=[(c.id,c.text) for c in chunks if c.kind=='passage']
    return hashlib.sha256((VERSION+json.dumps(entries,ensure_ascii=False)).encode()).hexdigest()

def apply_audit(chunks,report):
    leaves={c.id:c for c in chunks if c.kind=='passage'}
    if report.get('fingerprint')!=fingerprint(chunks):raise ValueError('Source audit is stale')
    rows=report['items']
    if len(rows)!=len(leaves) or {x['chunk_id'] for x in rows}!=set(leaves):raise ValueError('Incomplete source audit')
    spans=[]
    for row in rows:
        c=leaves[row['chunk_id']]
        for quote in row['instruction_quotes']:
            if len(quote)<4 or quote not in c.text:raise ValueError('Audit quote absent from original')
            start=c.start+c.text.index(quote);spans.append((c.doc_id,start,start+len(quote),quote))
    for c in chunks:
        c.instruction_spans=[{'start':a,'end':b,'text':q,'reason':'model_instruction_classifier'} for d,a,b,q in spans if d==c.doc_id and a<c.end and b>c.start]
    return {'checked_passages':len(rows),'instruction_spans':len(spans),'model':report.get('model',''),'fingerprint':report['fingerprint']}

def load_audit(chunks,cache_dir):
    name='source-audit-'+fingerprint(chunks)+'.json'
    for path in [Path(cache_dir)/name,Path(__file__).resolve().parent/name]:
        if path.exists():
            try:return apply_audit(chunks,json.loads(path.read_text(encoding='utf8')))
            except (ValueError,KeyError,TypeError):continue
    return None

def classify(chunks,api,cache_dir):
    leaves=[c for c in chunks if c.kind=='passage'];rows=[]
    part_dir=Path(cache_dir)/'source-audit-parts';part_dir.mkdir(parents=True,exist_ok=True)
    def batch(group):
        class BatchAudit(SourceAudit):
            items:list[SourceClassification]=Field(min_length=len(group),max_length=len(group))
        payload={'chunks':[{'chunk_id':c.id,'text':c.text} for c in group]}
        key=hashlib.sha256((VERSION+json.dumps(payload,ensure_ascii=False)).encode()).hexdigest()
        path=part_dir/(key+'.json');originals={c.id:c.text for c in group}
        def valid(items):
            return len(items)==len(group) and {x['chunk_id'] for x in items}==set(originals) and all(len(q)>=4 and q in originals[x['chunk_id']] for x in items for q in x['instruction_quotes'])
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf8'))
            if valid(cached):return cached
        for attempt in range(2):
            result=api.structured(PROMPT+'\nВерни ровно '+str(len(group))+' объектов, включая фрагменты без инструкций.',payload,BatchAudit)
            if isinstance(result,SourceAudit):
                items=[x.model_dump() for x in result.items]
                if valid(items):
                    path.write_text(json.dumps(items,ensure_ascii=False),encoding='utf8');return items
            payload['correction']='Предыдущий ответ неполон или содержит неточную цитату. Верни все исходные chunk_id; цитаты копируй точно.'
        raise ModelError('Incomplete instruction classification')
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(batch,leaves[i:i+8]) for i in range(0,len(leaves),8)]
        for future in as_completed(futures):rows.extend(future.result())
    rows.sort(key=lambda x:x['chunk_id'])
    report={'version':VERSION,'model':api.s.model,'fingerprint':fingerprint(chunks),'items':rows}
    summary=apply_audit(chunks,report)
    path=Path(cache_dir)/('source-audit-'+report['fingerprint']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)
    return summary,path
