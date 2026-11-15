"""Freeze a NEW proxy experiment from available historical answers, not missing v8 data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def prepare(source,consolidated,output,selected_ids=None,systems=None):
    cases={c['id']:c for c in map(json.loads,(source/'cases_effective.jsonl').read_text(encoding='utf8').splitlines())}
    pilot=json.loads((consolidated/'pilot/results.json').read_text(encoding='utf8'))
    ids=selected_ids or [r['id'] for r in pilot]
    systems=systems or ('vector14','coverage2257','review29','conditional35','batch_deadline_v8')
    answers={name:{} for name in systems}
    historical=[json.loads(line) for line in (consolidated/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    for row in historical:
        if row['run_id'] in systems and row['original_id'] in ids:
            case=cases[row['original_id']]
            if row['question']!=case['question']:raise ValueError('Historical question mismatch')
            if row['original_id'] in answers[row['run_id']]:raise ValueError('Duplicate historical answer')
            answers[row['run_id']][row['original_id']]=row['answer']
    for row in pilot:
        if row['question']!=cases[row['id']]['question']:raise ValueError('Pilot question mismatch')
        if 'batch_deadline_v8' in answers and row['id'] in ids:
            answers['batch_deadline_v8'][row['id']]=row['answer']
    if any(set(x)!=set(ids) for x in answers.values()):raise ValueError('Incomplete common cohort')
    output.mkdir(parents=True,exist_ok=False)
    # Extract only the known four imports required to build donor lexical inputs.
    # No frozen solution entry point is executed, and no model client is called.
    with tempfile.TemporaryDirectory() as directory:
        donor=Path(directory)
        archive=source/'frozen/conditional35.zip'
        if sha(archive)!='aaff1431120e42c4612244a0349bb1ee57a89ab83cbf3a808ef8e263e3db83b1':
            raise ValueError('Donor artifact changed')
        with zipfile.ZipFile(archive) as z:
            for name in ('legal_core.py','rag_safety.py','hallugraph.py','rag_telemetry.py'):
                (donor/name).write_bytes(z.read(name))
        sys.path.insert(0,str(donor))
        from legal_core import LegalAssistant, focused_search_question
        from rag_safety import clean_data
        from types import SimpleNamespace
        engine=LegalAssistant(source/'knowledge-base.zip',SimpleNamespace(base='',key='',rerank_model=''),use_embeddings=False)
        documents=[engine.source_records[s.id] for s in engine.sources]
        inputs=[]
        for identifier in ids:
            case=cases[identifier]
            query=focused_search_question(clean_data(case['question'])[0])
            indices=engine._lexical(query)[:64]
            inputs.append({'id':identifier,'family_id':case['family_id'],'question':case['question'],
                'golden':case['reference_answer'],'query':query,
                'retrieval_candidates':[documents[i] for i in indices],
                'rerank_texts':[engine.search_texts[i] for i in indices]})
        sys.path.pop(0)
    controls=[
        {'id':'same','question':'Какова цена по договору?','golden':'Цена 450 рублей (п. 1).',
         'baseline':'Цена 450 рублей (п. 1).','candidate':'Цена 450 рублей (п. 1).','expect':'tie'},
        {'id':'wrong-number','question':'Какова цена по договору?','golden':'Цена 450 рублей (п. 1).',
         'baseline':'Цена 450 рублей (п. 1).','candidate':'Цена 900 рублей (п. 1).','expect':'zero'},
        {'id':'needless-question','question':'Какова цена по договору?','golden':'Цена 450 рублей (п. 1).',
         'baseline':'Цена 450 рублей (п. 1).','candidate':'Уточните, пожалуйста, цену договора.','expect':'loss'},
        {'id':'correct-answer','question':'Какова цена по договору?','golden':'Цена 450 рублей (п. 1).',
         'baseline':'Цена 900 рублей (п. 1).','candidate':'Цена 450 рублей (п. 1).','expect':'win'}]
    write(output/'cases.json',inputs);write(output/'answers.json',answers);write(output/'controls.json',controls)
    observations=json.loads((consolidated/'platform-observations.json').read_text(encoding='utf8'))
    write(output/'historical-platform.json',observations)
    manifest={'version':'psb-platform-proxy-20261115','parent_code_commit':'f19a327',
        'status':'FROZEN_NOT_RUN','systems':list(systems),'cases':ids,'judge':'deepseek-v4.1-flash',
        'baseline':'fresh lexical top64 -> Qwen3-Reranker-8B top6 -> DeepSeek; no golden or labels',
        'judge_payload':'question, golden, anonymized A/B; no system identity or platform score',
        'scope':'8 previously exposed diagnostic cases shared with saved v8 pilot; not a representative or held-out benchmark',
        'source_files':{str(p.relative_to(source)):sha(p) for p in [source/'cases_effective.jsonl',source/'knowledge-base.zip',archive]},
        'input_hashes':{n:sha(output/n) for n in ('cases.json','answers.json','controls.json','historical-platform.json')},
        'code_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},
        'api_limits':{'chat_calls':100,'rerank_calls':8,'conservative_chat_token_ceiling':1000000},
        'differences_from_psb':['Hidden baseline, questions, golden and judge prompt unknown.',
         'Local historical outputs are not organizer outputs; runtime configurations differ.',
         'Golden-only judge cannot fully verify novel facts against the entire corpus.',
         'Either-round fabrication zero rule is an explicit local interpretation.',
         'Literal-gold stress zeroes unsupported extras and intentionally violates FAQ; sensitivity analysis only.'],
        'platform_score_prediction':False,'paid_calls_at_freeze':0}
    write(output/'freeze.json',manifest)
    print(json.dumps({'prepared':True,'cases':len(ids),'systems':len(systems),'paid_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--consolidated',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source.resolve(),a.consolidated.resolve(),a.output.resolve())
