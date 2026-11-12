"""Concurrent, resumable A/B judge using the same frozen rubric as judge.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark import ROOT,cases,corpus,read_answers,aggregate_scores
from judge import PROMPT,SOURCE_POLICY,complete,combine,payload_for_case,pair_payload
from context_export import load_contexts
from labels import corrected_cases, label_hash


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--services',type=Path,required=True);p.add_argument('--model',default='qwen3.8-max')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',default='development')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--ids')
    args=p.parse_args()
    secrets=json.loads(args.services.read_text());config=(secrets['LLM_BASE_URL'],secrets['LLM_API_KEY'],args.model)
    rows=corrected_cases(args.split)
    if args.ids:
        ids=args.ids.split(',');rows=[r for r in rows if r['id'] in ids]
        if set(ids)!={r['id'] for r in rows}:raise ValueError('Unknown IDs')
    ids=[r['id'] for r in rows]
    answers=read_answers(args.candidate,ids);baseline=read_answers(args.baseline,ids);docs=corpus()
    candidate_contexts,candidate_context_hash=load_contexts(args.candidate.parent)
    baseline_contexts,baseline_context_hash=load_contexts(args.baseline.parent)
    meta={'judge_model':args.model,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
          'source_policy':SOURCE_POLICY,
          'candidate_context_sha256':candidate_context_hash,'baseline_context_sha256':baseline_context_hash,
          'judge_implementation_sha256':digest(ROOT/'judge.py'),
          'runner_implementation_sha256':digest(Path(__file__)),
          'candidate_sha256':digest(args.candidate),'baseline_sha256':digest(args.baseline),
          'dataset_sha256':digest(ROOT/'cases.jsonl'),'corpus_sha256':digest(ROOT/'knowledge-base.zip'),
          'label_errata_sha256':label_hash(),
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
        payload=payload_for_case(row,docs,baseline[row['id']],answers[row['id']])
        item={'id':row['id'],'family_id':row['family_id'],'split':row['split']}
        item['evidence_paths']=sorted(payload['source_documents'])
        first_payload=pair_payload(payload,baseline[row['id']],answers[row['id']],baseline_contexts[row['id']],candidate_contexts[row['id']])
        second_payload=pair_payload(payload,answers[row['id']],baseline[row['id']],candidate_contexts[row['id']],baseline_contexts[row['id']])
        item['payload_sha256']=hashlib.sha256(json.dumps(first_payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        start=time.perf_counter()
        rounds=[]
        try:
            first=complete(first_payload,config)
            rounds.append(first)
            second=complete(second_payload,config)
            rounds.append(second)
            item.update(combine(first,second))
        except Exception as exc:
            item.update(status='judge_error',error_type=type(exc).__name__)
            safe_errors={'Fabricated claim must quote the actual answer','Invented proof quote',
                         'Invalid proof source','Missing literal proof','Incomplete judge output',
                         'Score contradicts a one-sided confirmed fabrication','Preference without a material diagnostic',
                         'Preference contradicts material diagnostics','Invalid preference labels'}
            if str(exc) in safe_errors:item['validation_error']=str(exc)
        item['rounds']=rounds
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
