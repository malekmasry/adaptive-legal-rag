"""Small sanity checks for the evaluator, independent of version ranking."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from benchmark import cases, corpus, read_answers
from judge import PROMPT, SOURCE_POLICY, complete, combine, payload_for_case, pair_payload
from context_export import load_contexts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--services',type=Path,required=True)
    parser.add_argument('--model',default='qwen3.8-max')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args()
    secrets=json.loads(args.services.read_text())
    config=(secrets['LLM_BASE_URL'],secrets['LLM_API_KEY'],args.model)
    rows={row['id']:row for row in cases()};docs=corpus()
    actual_contexts,_=load_contexts(args.runs/'vector14')
    actual_answers=read_answers(args.runs/'vector14/submission.csv',['S01c'])
    controls=[('same_correct','S01a',None),
              ('false_amount','S02a','Арендодатель вправе повышать плату на 9% каждый квартал, уведомив арендатора по телефону.'),
              ('missing_conditions','S10a','Заявку отправляют по электронной почте.'),
              ('optional_extra','S01a',rows['S01a']['reference_answer']+' По п. 2.1 помещение передаётся по акту не позднее 15 августа 2024 года.'),
              ('scoped_absence','S01c',actual_answers['S01c'])]

    def evaluate(control):
        name,key,candidate=control;row=rows[key];baseline=row['reference_answer']
        if name=='scoped_absence':baseline=actual_answers[key]
        if candidate is None:candidate=baseline
        payload=payload_for_case(row,docs,baseline,candidate)
        context=(actual_contexts[key] if name=='scoped_absence' else {'available':True,'sources':[
            {'id':'control_'+str(i),'path':path,'text':text} for i,(path,text) in enumerate(payload['source_documents'].items())]})
        try:
            first=complete(pair_payload(payload,baseline,candidate,context,context),config)
            second=complete(pair_payload(payload,candidate,baseline,context,context),config)
        except Exception as exc:
            return {'control':name,'id':key,'status':'judge_error','score':None,
                    'passed':False,'error_type':type(exc).__name__}
        item={'control':name,'id':key,**combine(first,second),'rounds':[first,second]}
        item['passed']=item['status']=='ok' and (
            (name in ('same_correct','optional_extra','scoped_absence') and item['score']==.5 and not item['confirmed_hallucination']) or
            (name=='false_amount' and item['score']==0 and item['confirmed_hallucination']) or
            (name=='missing_conditions' and item['score']<.5 and not item['confirmed_hallucination']))
        return item
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(evaluate,item) for item in controls]):
            result=future.result();results.append(result)
            print(json.dumps({k:result[k] for k in ('control','status','score','passed')}),flush=True)
            args.output.write_text(json.dumps({'complete':len(results)==len(controls),'controls':results},ensure_ascii=False,indent=2)+'\n')
    report={'judge_model':args.model,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
            'source_policy':SOURCE_POLICY,'all_passed':all(r['passed'] for r in results),'controls':results}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not report['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
