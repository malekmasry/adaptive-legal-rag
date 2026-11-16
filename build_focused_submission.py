"""Flat PSB artifact, without questions, credentials or benchmark answers."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent/'focused-rag'
MODULES=['question_brief.py','style_renderer.py','citation_labels.py','intent_rules.py','decision_index.py','decision_catalog.json','decision_checks.py','solution.py','pipeline.py','focused_context.py','relation_scope.py','relation_scope.json','answer_guides.py','corpus_playbook.py','corpus_playbook.json','rule_drafts.py','settings.py','model_api.py',
         'corpus.py','retrieval.py','security.py','diagnostics.py','schemas.py',
         'prompt_config.py','token_budget.py','requirements.txt','aliases.yaml',
         'relations.yaml','prompts.yaml','cl100k_base.tiktoken']

def build(output):
    names=MODULES+[p.name for p in ROOT.glob('vectors-*.npz')]
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None and z.namelist()==names
        assert all('/' not in n and '\\' not in n for n in names)
    assert output.stat().st_size < 128*1024*1024
    return hashlib.sha256(output.read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    print(build(a.output))
