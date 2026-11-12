"""One fixed retry for every unresolved judgment; never retry a valid low score."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from benchmark import ROOT, cases, corpus, read_answers
from judge import PROMPT, SOURCE_POLICY, complete, combine, payload_for_case, pair_payload
from context_export import load_contexts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--services',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    secrets=json.loads(args.services.read_text());docs=corpus()
    versions=('vector14','coverage2257','review29','conditional35')
    original,metadata,answers,contexts={}, {}, {}, {};jobs=[]
    for version in versions:
        path=args.input/(version+'.jsonl');meta=json.loads(path.with_suffix('.meta.json').read_text())
        if meta['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest() or meta['judge_implementation_sha256']!=digest(ROOT/'judge.py') or meta['source_policy']!=SOURCE_POLICY:
            raise ValueError('Judge changed; cannot silently retry with different instructions')
        items={row['id']:row for row in map(json.loads,path.read_text().splitlines())}
        if len(items)!=len(meta['ids']) or set(items)!=set(meta['ids']):raise ValueError('Original evaluation is incomplete')
        candidate=args.runs/version/'submission.csv';baseline=args.runs/'vector14'/'submission.csv'
        if digest(candidate)!=meta['candidate_sha256'] or digest(baseline)!=meta['baseline_sha256']:raise ValueError('Answers changed')
        answers[version]=read_answers(candidate,meta['ids']);original[version]=items
        contexts[version],context_hash=load_contexts(candidate.parent)
        if context_hash!=meta['candidate_context_sha256']:raise ValueError('Retrieved context changed')
        metadata[version]={**meta,'parent_judgments_sha256':digest(path),
            'retry_policy':'Exactly one two-order retry of every non-ok case; valid scores are never retried.',
            'retry_implementation_sha256':digest(Path(__file__)),
            'prior_unresolved_ids':[key for key,row in items.items() if row['status']!='ok']}
        rows={row['id']:row for row in cases(meta['split'])}
        jobs.extend((version,rows[key]) for key,row in items.items() if row['status']!='ok')
    baseline=answers['vector14']
    retries={version:{} for version in versions}

    def evaluate(job):
        version,row=job;key=row['id'];meta=metadata[version]
        config=(secrets['LLM_BASE_URL'],secrets['LLM_API_KEY'],meta['judge_model'])
        payload=payload_for_case(row,docs,baseline[key],answers[version][key])
        item={'id':key,'family_id':row['family_id'],'split':row['split'],
              'evidence_paths':sorted(payload['source_documents'])};rounds=[];start=time.perf_counter()
        try:
            rounds.append(complete(pair_payload(payload,baseline[key],answers[version][key],contexts['vector14'][key],contexts[version][key]),config))
            rounds.append(complete(pair_payload(payload,answers[version][key],baseline[key],contexts[version][key],contexts['vector14'][key]),config))
            item.update(combine(*rounds))
        except Exception as exc:item.update(status='judge_error',error_type=type(exc).__name__)
        item.update(rounds=rounds,elapsed_ms=round((time.perf_counter()-start)*1000,2))
        return version,item

    with (args.output/'retry_attempts.jsonl').open('w',buffering=1) as log,ThreadPoolExecutor(max_workers=max(1,min(args.workers,4))) as pool:
        for future in as_completed([pool.submit(evaluate,job) for job in jobs]):
            version,item=future.result();retries[version][item['id']]=item
            log.write(json.dumps({'version':version,**item},ensure_ascii=False)+'\n')
            print(json.dumps({'version':version,'id':item['id'],'status':item['status'],'score':item.get('score')}),flush=True)
    for version in versions:
        output=args.output/(version+'.jsonl')
        with output.open('w') as stream:
            for key in metadata[version]['ids']:
                old=original[version][key];retry=retries[version].get(key)
                chosen=({**retry,'previous_attempt':old} if retry['status']=='ok' else {**old,'retry_attempt':retry}) if retry else old
                stream.write(json.dumps(chosen,ensure_ascii=False)+'\n')
        output.with_suffix('.meta.json').write_text(json.dumps(metadata[version],ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
