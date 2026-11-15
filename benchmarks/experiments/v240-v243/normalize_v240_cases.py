"""Pre-run typography correction, preserving the initial freeze and all facts."""
import hashlib,json,re,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;root=W/'v240-experiment'
assert not (root/'calls.jsonl').exists() and not (root/'generated').exists()
original=root/'initial-spacing-freeze';original.mkdir()
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['development.json','holdout.json','dataset-freeze.json']:shutil.copyfile(root/n,original/n)
for n in ['development.json','holdout.json']:
    rows=read(root/n)
    for row in rows:
        for key in ['question','golden']:
            text=row[key]
            text=re.sub(r'(?<=[А-Яа-яЁёA-Za-z])(?=\d)|(?<=\d)(?=[А-Яа-яЁёA-Za-z])',' ',text)
            assert re.sub(r'\s','',text)==re.sub(r'\s','',row[key])
            row[key]=text
    (root/n).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
freeze=read(root/'dataset-freeze.json');freeze['version']='v240-diagnostic-v2-before-any-run'
freeze.update(development_sha256=sha(root/'development.json'),holdout_sha256=sha(root/'holdout.json'),
    pre_run_revision={'initial_freeze_sha256':sha(original/'dataset-freeze.json'),'script_sha256':sha(Path(__file__)),'scope':'Whitespace only in question/golden text; no fact changes. Applied before any model call or product experiment.'})
(root/'dataset-freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Final dataset freeze:18 development +12 holdout; no calls made.')
