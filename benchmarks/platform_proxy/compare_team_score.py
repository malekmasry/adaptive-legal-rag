"""Protocol audit: request the team's score directly in both answer orders.

Does not alter the frozen legacy judge or regenerate product answers. Both
the saved stable candidate and the new one are judged under this same protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from judge import PROMPT,validate,combine,aggregate,payload
from run import cached
from prepare import write,sha
from resume_checked_candidate import CircuitServices

TEAM_PROMPT=PROMPT.replace(
 'a_score: 1 если A значительно лучше B, .75 немного лучше, .5 сопоставимы,\n.25 немного хуже, 0 значительно хуже. Не применяй обнуление к a_score: это сделает код.',
 'team_score — оценка ОТВЕТА КОМАНДЫ относительно BASELINE. Поле team_label указывает, какой ответ принадлежит команде: A или B. Другой ответ — baseline.\n'
 'В обоих раундах: 1 если ответ КОМАНДЫ значительно лучше baseline; .75 немного лучше; .5 сопоставимы; .25 немного хуже; 0 значительно хуже.\n'
 'Если команда обозначена B, всё равно выставляй оценку команды B, а не оценку A. Не инвертируй team_score из-за порядка. Не применяй обнуление: это сделает код.'
).replace('{"a_score":0.5,','{"team_score":0.5,')
assert TEAM_PROMPT!=PROMPT and 'a_score:' not in TEAM_PROMPT


def pair(root,api,key,question,golden,baseline,candidate):
    rounds=[]
    for order in (0,1):
        data=payload(question,golden,baseline,candidate,order)
        data['team_label']='B' if order==0 else 'A'
        def call(data=data,order=order):
            raw=api.chat('JUDGE',TEAM_PROMPT,data)
            score=raw.get('team_score')
            if type(score) not in (int,float) or score not in (0,.25,.5,.75,1):raise ValueError('Invalid team score')
            # Only normalize for the unchanged aggregate code, not for the LLM.
            decision={**raw,'a_score':1-score if order==0 else score}
            return validate(decision,data['A'],data['B'])
        result=cached(root,key+f'-{order}',call)
        rounds.append({'status':result['status'],'decision':result.get('value'),'error_type':result.get('error_type')})
    return {'rounds':rounds,**combine(rounds)}


def main(a):
    start=time.monotonic();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    stable=json.loads((a.stable/'all-answers.json').read_text())['focused_v5_resumed']
    candidate=json.loads((a.candidate/'all-answers.json').read_text())[a.name]
    cases=json.loads((a.candidate/'cases.json').read_text())
    base=json.loads((a.candidate/'baseline.json').read_text())
    assert sha(a.stable/'baseline.json')==sha(a.candidate/'baseline.json')
    stable_cases=json.loads((a.stable/'cases.json').read_text())
    assert [(c['id'],c['question'],c['golden']) for c in stable_cases]==[(c['id'],c['question'],c['golden']) for c in cases]
    freeze={'protocol':'team-relative score explicitly requested in both rounds',
      'candidate':a.name,'input_hashes':{'baseline':sha(a.candidate/'baseline.json'),'cases':sha(a.candidate/'cases.json'),
      'stable_answers':sha(a.stable/'all-answers.json'),'candidate_answers':sha(a.candidate/'all-answers.json')},
      'old_prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'new_prompt_sha256':hashlib.sha256(TEAM_PROMPT.encode()).hexdigest(),
      'scope':'Different judge protocol; compare the two candidates here, not this number against legacy scores.',
      'api_limits':{'chat_calls':90,'rerank_calls':0,'conservative_chat_token_ceiling':1500000}}
    write(root/'freeze.json',freeze);(root/'judge-prompt.txt').write_text(TEAM_PROMPT,encoding='utf8')
    api=CircuitServices(json.loads(a.services.read_text()),root,max_chat_calls=90,max_rerank_calls=0,max_chat_tokens=1500000)
    try:
        controls=[]
        for key,left,right,expected in [('tie','30 календарных дней.','30 календарных дней.','tie'),
          ('win','90 календарных дней.','30 календарных дней.','win'),('loss','30 календарных дней.','90 календарных дней.','loss')]:
            result=pair(root,api,'control-'+key,'В условии договора указан срок оплаты 30 календарных дней. Какой срок оплаты по этому условию?',
                        'Срок оплаты составляет 30 календарных дней.',left,right)
            v=result.get('score');passed=result['status']=='ok' and (v==.5 if expected=='tie' else v>.5 if expected=='win' else v<.5)
            controls.append({'id':key,'passed':passed,**result})
        write(root/'controls.json',controls)
        if not all(c['passed'] for c in controls):
            print(json.dumps({'controls_passed':False,'stopped_before_comparison':True}),flush=True);return 2
        systems={'focused_v5_resumed':stable,a.name:candidate}
        jobs=[(name,c) for name in systems for c in cases]
        def work(job):
            name,c=job
            result=pair(root,api,name+'-'+c['id'],c['question'],c['golden'],base[c['id']]['value']['answer'],systems[name][c['id']])
            print(json.dumps({'system':name,'id':c['id'],'score':result.get('score'),'status':result['status']}),flush=True)
            return {'system':name,'id':c['id'],**result}
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(work,jobs))
        write(root/'judgments.json',rows)
        common=set.intersection(*[{r['id'] for r in rows if r['system']==name and r['status']=='ok'} for name in systems])
        summary={name:{'score':aggregate([r for r in rows if r['system']==name]),
           'common_score':aggregate([r for r in rows if r['system']==name and r['id'] in common]),
           'fabrication_zero_count':sum(r.get('fabrication_zero',False) for r in rows if r['system']==name)} for name in systems}
        calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()]
        done=[r for r in calls if r['event']=='completed']
        report={'systems':summary,'common_ids':sorted(common),'seconds':round(time.monotonic()-start,3),
          'controls_passed':True,'protocol':freeze,'usage':{'deepseek_calls':len(done),
          'deepseek_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in done),
          'api_errors':sum(r.get('status')!='ok' for r in done),'qwen_calls':0}}
        write(root/'summary.json',report);print(json.dumps(report),flush=True)
    finally:api.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('stable','candidate','output','services'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--name',required=True);a=p.parse_args();raise SystemExit(main(a))
