"""Compact diagnostics, without altering or retrying any judgment."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'v240-experiment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','holdout'],default='development');a=p.parse_args()
if a.phase=='holdout':assert (ROOT/'selection.json').exists()
cases={c['id']:c for c in read(ROOT/(a.phase+'.json'))}
coverage=read(ROOT/(a.phase+'-coverage.json'));answers=read(ROOT/(a.phase+'-answers.json'))
rows=read(ROOT/(a.phase+'-judgments.json')) if (ROOT/(a.phase+'-judgments.json')).exists() else []
lookup={(r['system'],r['id']):r for r in rows}
for name,values in coverage.items():
    anchors=[s for row in values for s in row.get('anchors',[])]
    print(name,{'complete_evidence_cases':sum(r.get('all_anchors_present') is True for r in values),'cases_with_sources':sum(r.get('all_anchors_present') is not None for r in values),
        'present_anchors':sum(s['present'] for s in anchors),'required_anchors':len(anchors),'ready_drafts':sum(r.get('draft_origin')=='source_rules' for r in values),
        'logical_model_calls':sum(r.get('logical_model_calls',0) for r in values),'omitted_document_cases':sum(bool(r.get('omitted_documents')) for r in values)})
    for row in values:
        if row.get('all_anchors_present') is False:print('MISSING',name,row['id'],[s['filename'] for s in row['anchors'] if not s['present']])
for ident,c in cases.items():
    print('\nCASE',ident,c['category'],'\nQUESTION',c['question'],'\nEXPECTED',c['golden'])
    print('BASELINE',answers['v239_'+a.phase][ident])
    for name,vals in answers.items():
        if name=='v239_'+a.phase:continue
        variant=name.split('_')[0];r=lookup.get((variant+'_vs_v239',ident),{})
        text=vals[ident]
        print(variant,'score',r.get('score'),'SAME' if text==answers['v239_'+a.phase][ident] else text)
        if r.get('status')!='ok' or r.get('score')!=.5 or r.get('fabrication_zero'):
            print('JUDGE',json.dumps(r,ensure_ascii=False))
