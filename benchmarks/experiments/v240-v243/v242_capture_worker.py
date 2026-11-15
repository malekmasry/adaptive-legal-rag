"""External observation harness; normal mode changes no model input/output.

Oracle mode is diagnostic only. It substitutes manually selected source spans
for retrieval/context; no golden, answer or quality score is available here.
"""
import csv,hashlib,json,os,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from pipeline import Pipeline
from question_brief import resolved_paragraphs
from solution import run
from focused_context import source_name,render_paragraphs,date_note,search_queries
from citation_labels import finish_answer
from token_budget import estimate_tokens
import pipeline as pipeline_module

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (Path(os.environ['DATA_DIR'])/'questions.csv').open(encoding='utf8',newline='') as f:questions={r['question']:r['id'] for r in csv.DictReader(f)}
assert len(questions)==len(set(questions.values()))
capture_dir=ROOT/'captured';capture_dir.mkdir(exist_ok=True)
state=threading.local();core=Pipeline();original_ask=core.ask;original_structured=core.api.structured
mode=os.environ.get('EXPERIMENT_MODE','observe');assert mode in ('observe','oracle')
if mode=='oracle':
    sources=json.loads((ROOT/'oracle_sources.json').read_text(encoding='utf8'))
    by_name={d.filename:d for d in core.docs.values()}
    def oracle_search(plan,**kwargs):
        spans=sources[state.ident];docs=list(dict.fromkeys(x['filename'] for x in spans))
        chunks=[next(c for c in core.all_chunks if c.doc_id==by_name[n].id and c.kind=='passage') for n in docs]
        return chunks,{'mode':'manual_source_oracle','selected':[c.id for c in chunks],'degraded':[],'candidates':[],'seconds':0}
    def oracle_bundle(chunks,documents,relations,budget,question=''):
        evidence=[];used=0
        for s in sources[state.ident]:
            d=by_name[s['filename']];assert hashlib.sha256(d.raw.encode()).hexdigest()==s['sha256']
            text=d.raw[s['start']:s['end']];assert hashlib.sha256(text.encode()).hexdigest()==s['text_sha256']
            item={'source_id':'D'+str(len(evidence)+1),'document':source_name(d),'title':d.title,
                  'scope':'полный документ' if s['full_document'] else f"фрагмент, символы {s['start']}:{s['end']}",'text':text}
            used+=estimate_tokens(json.dumps(item,ensure_ascii=False));assert used<=budget,'Oracle exceeds same evidence budget'
            evidence.append(item)
        return evidence,{'estimated_tokens':used,'budget':budget,'relation_decisions':[],
            'primary_documents':list(dict.fromkeys(s['filename'] for s in sources[state.ident])),
            'omitted_documents':[],'full_documents':sum(e['scope']=='полный документ' for e in evidence),'manual_source_oracle':True}
    core.retriever.search=oracle_search
    core.decision_index.select=lambda *a,**k:([],{'selected':[],'tokens':0,'manual_source_oracle':True})
    pipeline_module.context_bundle=oracle_bundle
    core.initialize=lambda:None

def capture(instruction,payload,schema):
    record={'schema':schema.__name__,'instruction':instruction,'payload':json.loads(json.dumps(payload,ensure_ascii=False))}
    state.calls.append(record);start=time.monotonic()
    try:
        result=original_structured(instruction,payload,schema)
        record['response']=result.model_dump();return result
    except Exception as exc:
        record['error_type']=type(exc).__name__;raise
    finally:record['seconds']=time.monotonic()-start
core.api.structured=capture

def ask(question,**kwargs):
    state.ident=questions[question];state.calls=[];start=time.monotonic();result=None
    try:
        result=original_ask(question,**kwargs);return result
    finally:
        initial=None;payload=None
        for call in state.calls:
            p=call['payload']
            if 'previous_draft' in p and 'format_error' not in p:initial=p['previous_draft'];payload=p;break
            if initial is None and call.get('response',{}).get('paragraphs'):initial=call['response'];payload=p
        initial_answer=None
        if initial and payload:
            try:
                saved=SimpleNamespace(paragraphs=[SimpleNamespace(model_dump=lambda p=p:p) for p in initial['paragraphs']])
                text,refs=render_paragraphs(resolved_paragraphs(saved,payload.get('citation_catalog',[])),payload['evidence'])
                text=date_note(question,text,core.s.as_of,bool(refs))
                initial_answer=finish_answer(text,search_queries(question)[0],refs,payload['evidence'],core.docs)
            except ValueError:pass
        write(capture_dir/(state.ident+'.json'),{'id':state.ident,'question':question,'mode':mode,
            'calls':state.calls,'initial_answer':initial_answer,'result':result,'seconds':time.monotonic()-start})
core.ask=ask
report=run(output_path=Path(os.environ['OUTPUT_CSV']),pipeline=core)
print(json.dumps(report,ensure_ascii=False),flush=True)
core.close()
