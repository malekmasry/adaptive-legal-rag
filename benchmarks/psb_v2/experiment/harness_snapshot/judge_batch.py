"""Concurrent, resumable A/B judge using the same frozen rubric as judge.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark import ROOT,cases,corpus,read_answers,aggregate_scores
from judge import PROMPT,complete,combine


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_for(row,docs):
    selected={e['path'] for e in row['evidence']}
    payload={k:row[k] for k in ['question','reference_answer','required_facts','critical_errors','as_of','event_date','answerability']}
    payload['source_documents']={p:docs[p] for p in sorted(selected)}
    payload['catalog']=sorted(docs)
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--services',type=Path,required=True);p.add_argument('--model',default='qwen3.8-max')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',default='development')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--ids')
    args=p.parse_args()
    secrets=json.loads(args.services.read_text());config=(secrets['LLM_BASE_URL'],secrets['LLM_API_KEY'],args.model)
    rows=cases(args.split)
    if args.ids:
        ids=args.ids.split(',');rows=[r for r in rows if r['id'] in ids]
        if set(ids)!={r['id'] for r in rows}:raise ValueError('Unknown IDs')
    ids=[r['id'] for r in rows]
    answers=read_answers(args.candidate,ids);baseline=read_answers(args.baseline,ids);docs=corpus()
    meta={'judge_model':args.model,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
          'judge_implementation_sha256':digest(ROOT/'judge.py'),
          'candidate_sha256':digest(args.candidate),'baseline_sha256':digest(args.baseline),
          'dataset_sha256':digest(ROOT/'cases.jsonl'),'corpus_sha256':digest(ROOT/'knowledge-base.zip'),
          'ids':ids,'split':args.split,'organizer_equivalence':False,
          'candidate_run':json.loads((args.candidate.parent/'run.meta.json').read_text()),
          'baseline_run':json.loads((args.baseline.parent/'run.meta.json').read_text())}
    for role in ('candidate','baseline'):
        run=meta[role+'_run']
        if run.get('answers_sha256')!=meta[role+'_sha256'] or run.get('dataset_sha256')!=meta['dataset_sha256'] or run.get('corpus_sha256')!=meta['corpus_sha256']:
            raise ValueError('Mismatched run provenance')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    meta_path=args.output.with_suffix('.meta.json')
    if meta_path.exists() and json.loads(meta_path.read_text())!=meta:raise ValueError('Cannot mix different runs')
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    results=[json.loads(s) for s in args.output.read_text().splitlines()] if args.output.exists() else []
    done={r['id'] for r in results}
    if len(done)!=len(results) or not done<=set(ids):raise ValueError('Invalid resume file')

    def evaluate(row):
        payload=payload_for(row,docs)
        item={'id':row['id'],'family_id':row['family_id'],'split':row['split']}
        start=time.perf_counter()
        try:
            first=complete({**payload,'A':baseline[row['id']],'B':answers[row['id']]},config)
            second=complete({**payload,'A':answers[row['id']],'B':baseline[row['id']]},config)
            item.update(combine(first,second));item['rounds']=[first,second]
        except Exception as exc:item.update(status='judge_error',error_type=type(exc).__name__)
        item['elapsed_ms']=round((time.perf_counter()-start)*1000,2)
        return item

    with args.output.open('a',encoding='utf-8',buffering=1) as out,ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        futures=[pool.submit(evaluate,row) for row in rows if row['id'] not in done]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            out.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({'id':result['id'],'status':result['status'],'score':result.get('score'),'finished':len(results),'total':len(rows)}),flush=True)
    summary=aggregate_scores(results)
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
