"""Local PSB evaluation. Offline commands use Python's standard library only.

See README.md. validate, retrieval and score are offline diagnostics.
Use run_matrix.py for pipeline execution; direct generation is not exposed here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent

def cases(split="all"):
    rows = [json.loads(line) for line in (ROOT/"cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if split=="all" or r["split"]==split]

def corpus():
    with ZipFile(ROOT/"knowledge-base.zip") as z:
        return {name:z.read(name).decode("utf-8-sig") for name in z.namelist() if name.endswith(".txt")}

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def validate():
    rows=cases(); docs=corpus()
    manifest=json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    assert len(rows)==manifest["cases"] and len({r['id'] for r in rows})==len(rows)
    assert hashlib.sha256((ROOT/"knowledge-base.zip").read_bytes()).hexdigest()==manifest['archive_sha256']
    assert hashlib.sha256((ROOT/"cases.jsonl").read_bytes()).hexdigest()==manifest['dataset_sha256']
    for d in manifest["documents"]:
        assert hashlib.sha256(docs[d['path']].encode()).hexdigest()==d['sha256_utf8']
    family_splits=defaultdict(set)
    for row in rows:
        family_splits[row['family_id']].add(row['split'])
        assert row['question'].strip() and row['reference_answer'].strip()
        assert row['required_facts']
        if row['answerability']=='answerable': assert row['evidence']
        for e in row['evidence']:
            assert 0<=e['start']<e['end']<=len(docs[e['path']])
            assert docs[e['path']][e['start']:e['end']]==e['text']
    for family,splits in family_splits.items():
        assert not ('holdout' in splits and len(splits)>1), (family,splits)
    print(json.dumps({'validation':'passed','cases':len(rows),'documents':len(docs),
          'splits':manifest['splits'],'review':'independent expert review still required'},ensure_ascii=False))

def merge_ranges(ranges):
    result=[]
    for a,b in sorted(ranges):
        if a>=b: continue
        if result and a<=result[-1][1]: result[-1]=(result[-1][0],max(b,result[-1][1]))
        else: result.append((a,b))
    return result

def overlap(a,b,ranges):
    return sum(max(0,min(b,d)-max(a,c)) for c,d in merge_ranges(ranges))

def retrieval_metrics(row, contexts, docs):
    # Gold offsets are independent of any particular chunk IDs or boundaries.
    selected=defaultdict(list)
    for x in contexts:
        path=x['path'];a=x['start'];b=x['end']
        if path not in docs or not (type(a) is int and type(b) is int and 0<=a<b<=len(docs[path])):
            raise ValueError('Invalid context span')
        if x.get('text') != docs[path][a:b]:
            raise ValueError('Context text does not match original offsets')
        selected[path].append((a,b))
    gold=defaultdict(list)
    for e in row['evidence']:gold[e['path']].append((e['start'],e['end']))
    gold={p:merge_ranges(v) for p,v in gold.items()}
    selected={p:merge_ranges(v) for p,v in selected.items()}
    count=lambda path,a,b:sum(not ch.isspace() for ch in docs[path][a:b])
    covered=lambda path,a,b:sum(count(path,max(a,c),min(b,d)) for c,d in selected.get(path,[]) if min(b,d)>max(a,c))
    total=sum(count(p,a,b) for p,v in gold.items() for a,b in v)
    context_size=sum(b-a for v in selected.values() for a,b in v)
    context_nonspace=sum(count(p,a,b) for p,v in selected.items() for a,b in v)
    hits=sum(covered(p,a,b) for p,v in gold.items() for a,b in v)
    misses=[i for i,e in enumerate(row['evidence']) if covered(e['path'],e['start'],e['end'])<count(e['path'],e['start'],e['end'])]
    rule_misses=[i for i in misses if row['evidence'][i].get('role')!='applicability']
    return {'gold_nonspace_char_recall':hits/total if total else None,
        'annotated_span_precision':hits/context_nonspace if total and context_nonspace else (0 if total else None),
        'all_evidence_present':not misses if total else None,
        'all_rule_evidence_present':not rule_misses if total else None,
        'missing_evidence_indices':misses,'context_characters':context_size}

def describe_source(s):
    return {'id':s.id,'path':s.path,'start':s.start,'end':s.end,'text':s.text}

def load_engine(args, use_services=False):
    repo=Path(args.repo).resolve()
    sys.path.insert(0,str(repo))
    from legal_core import LegalAssistant, ModelProxy
    proxy=ModelProxy()
    if not (use_services and args.rerank):proxy.rerank_model=''
    if use_services and not proxy.ready:
        raise ValueError('Generation requires configured LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (or aliases).')
    return LegalAssistant(ROOT/'knowledge-base.zip',proxy,use_embeddings=bool(use_services and args.embeddings))

def retrieve(args):
    from_import_repo=Path(args.repo).resolve()
    sys.path.insert(0,str(from_import_repo))
    from single_pass import clean_data,expand_context,pack_context
    engine=load_engine(args)
    docs=corpus();report=[]
    for row in cases(args.split):
        start=time.perf_counter()
        cleaned,_=clean_data(row['question'])
        sources=engine.retrieve(cleaned)
        records=pack_context(engine,expand_context(engine,sources))
        context=[describe_source(engine.by_id[r['id']]) for r in records]
        report.append({'id':row['id'],'family_id':row['family_id'],'category':row['category'],
            'split':row['split'], 'context':context,
            'context_metadata':[{k:r[k] for k in ['id','document','clause','revision'] if k in r} for r in records],
            **retrieval_metrics(row,context,docs),'elapsed_ms':round((time.perf_counter()-start)*1000,3)})
    scored=[r for r in report if r['all_evidence_present'] is not None]
    summary={'method':'existing lexical retrieval + structural expansion + context packing',
        'embeddings':False,'reranker':False,'answer_generation':False,
        'cases':len(report),'evidence_cases':len(scored),'all_evidence_present':sum(r['all_evidence_present'] for r in scored),
        'all_rule_evidence_present':sum(r['all_rule_evidence_present'] for r in scored),
        'mean_gold_nonspace_char_recall':statistics.mean(r['gold_nonspace_char_recall'] for r in scored) if scored else None,
        'mean_annotated_span_precision':statistics.mean(r['annotated_span_precision'] for r in scored) if scored else None,
        'note':'Span metrics ignore whitespace. Applicability may also be conveyed by metadata, which is not span-scored. Precision is against annotations, not all useful context. This is not answer accuracy.'}
    from legal_core import integer_setting
    configuration={'top_k':integer_setting('TOP_K',12,4,30),
        'context_max_chars':integer_setting('CONTEXT_MAX_CHARS',18000,2000,80000),
        'context_max_sources':integer_setting('CONTEXT_MAX_SOURCES',18,1,48),
        'dataset_sha256':hashlib.sha256((ROOT/'cases.jsonl').read_bytes()).hexdigest(),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(from_import_repo.glob('*.py'))}}
    write_json(args.output,{'summary':summary,'configuration':configuration,'cases':report})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def generate(args):
    os.environ['PIPELINE_MODE']=args.pipeline
    engine=load_engine(args,True)
    from legal_core import Source
    from single_pass import SinglePassPipeline
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    traces=output.with_suffix('.trace.jsonl')
    rows=cases(args.split)
    if args.oracle:
        rows=[r for r in rows if r['answerability']=='answerable']
    # Sequential by design: reproducible diagnosis rather than throughput testing.
    with output.open('w',encoding='utf-8',newline='') as f,traces.open('w',encoding='utf-8') as trace:
        writer=csv.writer(f);writer.writerow(['id','answer']);f.flush()
        for row in rows:
            if args.oracle:
                selected=[]
                for i,e in enumerate(row['evidence']):
                    document=next(d for d in engine.documents if d.path==e['path'])
                    anchor=next((s for s in engine.by_document[document.id] if s.start<=e['start']<s.end),engine.by_document[document.id][0])
                    sid=f'S{90000+i}'
                    source=Source(sid,document.id,document.title,document.path,anchor.clause,e['text'],e['start'],e['end'])
                    template=dict(engine.source_records[anchor.id])
                    template.update(id=sid,text=e['text'])
                    engine.by_id[sid]=source;engine.source_records[sid]=template;selected.append(source)
                result=SinglePassPipeline(engine).run(row['question'],sources_override=selected)
            else:
                result=engine.answer(row['question'])
            context_events=[e for e in result.diagnostics.get('events',[]) if e.get('stage')=='context' and 'source_ids' in e]
            context_ids=context_events[-1]['source_ids'] if context_events else []
            context=[describe_source(engine.by_id[s]) for s in context_ids if s in engine.by_id]
            writer.writerow([row['id'],result.text]);f.flush()
            trace.write(json.dumps({'id':row['id'],'status':result.status,'oracle':args.oracle,
                'context_available':bool(context_events),'context':context,'diagnostics':result.diagnostics},ensure_ascii=False)+'\n');trace.flush()
    print(json.dumps({'answers':str(output),'traces':str(traces),'cases':len(rows),'oracle_answerable_only':args.oracle}))

def read_answers(path, allowed_ids):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','answer']:raise ValueError('CSV columns must be exactly id,answer')
        result={}
        for row in reader:
            if set(row)!= {'id','answer'} or row['answer'] is None:raise ValueError('Malformed CSV row')
            key=row['id']
            if key in result:raise ValueError('Duplicate id: '+key)
            if not row['answer'].strip() or len(row['answer'])>10000:raise ValueError('Empty/oversized answer: '+key)
            result[key]=row['answer']
    missing=set(allowed_ids)-result.keys()
    if missing:raise ValueError('Missing answer IDs: '+','.join(sorted(missing)))
    # Superset is allowed so a single result file can be evaluated by split.
    return result

def aggregate_scores(rows):
    valid=[r for r in rows if r.get('status')=='ok']
    scores=[r['score'] for r in valid]
    if not scores:return {'score':None,'match_rate':None,'win_rate':None,'evaluated':0,'attempted':len(rows),'coverage':0}
    wins=sum(s>0.5 for s in scores);losses=sum(s<0.5 for s in scores)
    mr=statistics.mean(scores);wr=wins/(wins+losses) if wins+losses else 0.5
    return {'score':0.7*mr+0.3*wr,'match_rate':mr,'win_rate':wr,'wins':wins,'losses':losses,
        'ties':len(scores)-wins-losses,'evaluated':len(scores),'attempted':len(rows),'coverage':len(scores)/len(rows)}

def bootstrap_interval(rows, repetitions=2000):
    groups=defaultdict(list)
    for r in rows:
        if r.get('status')=='ok':groups[r['family_id']].append(r)
    keys=list(groups)
    if len(keys)<2:return None
    rng=random.Random(20261111);values=[]
    for _ in range(repetitions):
        sample=[r for k in rng.choices(keys,k=len(keys)) for r in groups[k]]
        values.append(aggregate_scores(sample)['score'])
    values.sort()
    return {'low':values[int(.025*repetitions)],'high':values[min(repetitions-1,int(.975*repetitions))],
        'method':'family-cluster percentile bootstrap','independent_families':len(keys),
        'warning':'Describes resampling variability of this local suite, not PSB score uncertainty.'}

def score(args):
    rows=[json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate result IDs')
    for r in rows:
        if r.get('status')=='ok' and (type(r.get('score')) not in (float,int) or r['score'] not in [i/8 for i in range(9)]):
            raise ValueError('Expected a two-round score in increments of 0.125')
    summary=aggregate_scores(rows);summary['bootstrap_95']=bootstrap_interval(rows)
    summary['label']='LOCAL pairwise proxy; organizer baseline, judge model and prompt are unknown'
    if args.output:write_json(args.output,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('validate')
    for name in ['retrieval']:
        q=sp.add_parser(name);q.add_argument('--repo',required=True);q.add_argument('--output',required=True)
        q.add_argument('--split',choices=['all','development','holdout','official_demo'],default='development')
        if name=='generate':
            q.add_argument('--pipeline',choices=['single_pass','adaptive'],default='single_pass')
            q.add_argument('--embeddings',action='store_true');q.add_argument('--rerank',action='store_true');q.add_argument('--oracle',action='store_true')
    q=sp.add_parser('score');q.add_argument('--input',required=True);q.add_argument('--output')
    args=p.parse_args()
    {'validate':lambda _:validate(),'retrieval':retrieve,'score':score}[args.command](args)

if __name__=='__main__':main()
