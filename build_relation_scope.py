"""Bind reviewed relationship topics and provision selectors to source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

ROOT=Path(__file__).resolve().parent/'focused-rag'
POLICIES={
 'amends':{'topics':['delivery_address','amendment_question'],
   'affected_provisions':['п. 3.2 договора П-17/2024'],
   'applicability':{'field':'specification_date','operator':'>','value':'2024-10-01',
     'note':'Keep the condition for both earlier and later specifications; delivery date is not the trigger.'},
   'source_selectors':['line:Изложить пункт 3.2','line:«3.2.','line:Новый адрес доставки','line:Остальные условия','line:Настоящее дополнительное соглашение'],
   'target_selectors':['3.2']},
 'supersedes':{'topics':['approval','approval_threshold','approval_form','approved_template','foreign_checks'],
   'affected_provisions':['редакция регламента целиком'],
   'applicability':{'field':'event_date','operator':'>=','value':'2025-03-01',
     'note':'Keep both editions when relevant; do not infer event date from a contract identifier.'},
   'source_selectors':None,'target_selectors':None},
 'supplements':{'topics':['foreign_checks'],
   'affected_provisions':['п. 5.4 регламента'],
   'applicability':{'field':'event_date','operator':'>=','value':'2025-04-21',
     'note':'The letter date is not the effective date; show the condition for before/after questions.'},
   'source_selectors':['line:С 21 апреля','line:Санкционную проверку','line:Требование применяется','line:Договоры с иностранными','line:Приложение 1.'],
   'target_selectors':['line:Редакция от','5.4']},
}

def build(archive):
    with zipfile.ZipFile(archive) as z:
        sources={n:z.read(n).decode('utf-8-sig') for n in z.namelist() if n.endswith('.txt')}
    rows=[]
    for old in yaml.safe_load((ROOT/'relations.yaml').read_text(encoding='utf8'))['relations']:
        if old['kind']=='cites':
            topic='tax_penalty' if 'ст114' in old['source'] else 'vat'
            policy={'topics':[topic],'affected_provisions':['отсылка к другой статье'],
              'applicability':{'type':'supporting_legal_reference'},'source_selectors':None,'target_selectors':None}
        else:policy=POLICIES[old['kind']]
        rows.append({**old,**policy,**{side+'_sha256':hashlib.sha256(sources[old[side]].encode()).hexdigest() for side in ('source','target')}})
    (ROOT/'relation_scope.json').write_text(json.dumps({'version':1,'relations':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'relations':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);build(p.parse_args().archive)
