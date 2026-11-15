"""Replay saved patch decisions and rendering; makes no model calls."""
import hashlib,json,sys,zipfile
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment'
runtime=ROOT/'generated/v245_regression'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
with zipfile.ZipFile(ROOT/'packages/v245.zip') as z:
    for info in z.infolist():
        if not info.is_dir():assert (runtime/info.filename).read_bytes()==z.read(info),info.filename
sys.path.insert(0,str(runtime))
from pipeline import Answer,Paragraph,Reference
from question_brief import answer_schema,resolved_paragraphs
from targeted_review import PatchReview,apply_patches
from focused_context import render_paragraphs,date_note,search_queries
from citation_labels import finish_answer
from corpus import ingest
docs={d.id:d for d in ingest(runtime/'data')[0]}
answers=read(ROOT/'v245-answers.json');rows=[]
for case in read(ROOT/'cases.json'):
    ident=case['id'];cap=read(runtime/'captured'/(ident+'.json'))
    review=next((c for c in cap['calls'] if c['schema']=='PatchReview'),None)
    call=review or next(c for c in reversed(cap['calls']) if c.get('response',{}).get('paragraphs'))
    payload=call['payload'];catalog=payload.get('citation_catalog',[])
    raw=payload['previous_draft'] if review else call['response']
    schema=answer_schema(Answer,Paragraph,Reference,payload['question_brief'],catalog) if all('part_id' in p for p in raw['paragraphs']) else Answer
    draft=schema.model_validate(raw);original=draft.model_dump()
    if review:
        revised,patch_report=apply_patches(draft,payload['evidence'],PatchReview.model_validate(review['response']),catalog)
        assert patch_report==cap['result']['trace']['patch_review'],ident
        assert draft.model_dump()==original,ident
    else:revised=draft;patch_report={'applied':0,'review_not_needed':True}
    text,refs=render_paragraphs(resolved_paragraphs(revised,catalog),payload['evidence'])
    text=date_note(cap['question'],text,payload['current_date'],bool(refs))
    text=finish_answer(text,search_queries(cap['question'])[0],refs,payload['evidence'],docs)
    assert text==answers[ident]==cap['result']['answer'],ident
    assert revised.status==cap['result']['status'],ident
    rows.append({'id':ident,'exact_final_answer':True,'draft_not_mutated':True,'patches':patch_report,
                 'answer_sha256':hashlib.sha256(text.encode()).hexdigest()})
result={'status':'passed','model_calls':0,'package_files_unchanged':True,'cases':len(rows),'rows':rows}
(ROOT/'replay-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
