"""Offline replay proving the final public-wording cleanup changes no sources.

No generation or judging calls. Reported LLM scores remain those of the frozen
v243; the single equivalent-wording output is explicitly not rejudged.
"""
import ast,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
W=Path(__file__).resolve().parent;R=W/'v242-experiment'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(R/'packages/v243.zip') as old,zipfile.ZipFile(R/'packages/v243-final.zip') as final:
    assert old.namelist()==final.namelist()
    assert [n for n in old.namelist() if old.read(n)!=final.read(n)]==['question_brief.py']
    a=ast.parse(old.read('question_brief.py'));b=ast.parse(final.read('question_brief.py'))
    def other(tree):return [ast.dump(n) for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='resolved_paragraphs']
    assert other(a)==other(b)
    inspect=R/'inspect-final';inspect.mkdir(exist_ok=True);final.extractall(inspect)
    node=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='resolved_paragraphs')
import re
scope={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-v243-renderer>','exec'),scope)
old_resolve=scope['resolved_paragraphs'];sys.path.insert(0,str(inspect))
from question_brief import resolved_paragraphs
from focused_context import render_paragraphs,date_note,search_queries
from citation_labels import finish_answer
from corpus import ingest
docs={d.id:d for d in ingest(R)[0]};proof=[];outputs={}
for phase in ['development','holdout']:
    outputs[phase]={};answers=read(R/('v243-'+phase+'-answers.json'))['v243_'+phase]
    for c in read(R/(phase+'.json')):
        saved=read(R/'generated'/('v242_development' if phase=='development' else 'v243_holdout')/'captured'/(c['id']+'.json'))
        value=payload=None
        for call in reversed(saved['calls']):
            response=call.get('response',{})
            if response.get('paragraphs'):value=response;payload=call['payload'];break
            if call['schema']=='DraftCheck' and 'keep_prepared' in response:
                value=call['payload']['previous_draft'] if response['keep_prepared'] else response['correction'];payload=call['payload'];break
        assert value and payload
        obj=SimpleNamespace(paragraphs=[SimpleNamespace(model_dump=lambda p=p:p) for p in value['paragraphs']])
        before=old_resolve(obj,payload.get('citation_catalog',[]));after=resolved_paragraphs(obj,payload.get('citation_catalog',[]))
        assert [p['citations'] for p in before]==[p['citations'] for p in after]
        def render(paragraphs):
            text,refs=render_paragraphs(paragraphs,payload['evidence'])
            return finish_answer(date_note(c['question'],text,'2026-11-15',bool(refs)),search_queries(c['question'])[0],refs,payload['evidence'],docs)
        expected=render(before);actual=render(after);assert expected==answers[c['id']]
        assert not re.search(r'(?m)^\s*P\d+\s*:|\bquestion_brief\b|\bcitation_id\b|\bevidence\b',actual)
        outputs[phase][c['id']]=actual
        proof.append({'phase':phase,'id':c['id'],'exact_pre_cleanup_reconstruction':True,'changed':actual!=expected,
            'references_unchanged':True,'before_sha256':hashlib.sha256(expected.encode()).hexdigest(),'after_sha256':hashlib.sha256(actual.encode()).hexdigest()})
assert [(p['phase'],p['id']) for p in proof if p['changed']]==[('holdout','H12')]
repo=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
result={'package_sha256':sha(R/'packages/v243-final.zip'),'tested_package_sha256':sha(R/'packages/v243.zip'),
    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    'script_sha256':sha(Path(__file__)),'unit_tests':83,'test_log_sha256':sha(W/'v243-final-tests.log'),
    'replayed':30,'changed_ids':['H12'],'generation_calls':0,'judge_calls':0,'fresh_full_live_run_of_final_zip':False,
    'scope':'Pure rendering cleanup: in evidence becomes in provided sources. Source references and legal facts unchanged. LLM scores are from frozen v243 before this one wording change; no new independent score claimed.',
    'proof':proof}
for name,data in [('final-rendering-verification.json',result),('final-answers.json',outputs)]:
    (R/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:result[k] for k in ['package_sha256','source_commit','unit_tests','replayed','changed_ids','generation_calls','judge_calls']}))
