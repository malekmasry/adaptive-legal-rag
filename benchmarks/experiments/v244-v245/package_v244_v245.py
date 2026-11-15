"""Archive every attempt and frozen source; recursive scan against private keys."""
import hashlib,io,json,subprocess,zipfile
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment';OUT=W.parent/'outputs'
REPO=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda data:hashlib.sha256(data).hexdigest()
assert (ROOT/'v244-summary.json').exists() and (ROOT/'v245-summary.json').exists()
assert read(ROOT/'manual-audit.json')['status']=='completed'
secrets=[]
def collect(obj):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if isinstance(value,str) and len(value)>=12 and any(s in key.lower() for s in ('key','token','secret','authorization')):secrets.append(value.encode())
            elif isinstance(value,(dict,list)):collect(value)
    elif isinstance(obj,list):
        for value in obj:collect(value)
for name in ('platform-proxy-services.private.json','local_api.json','siliconflow-reranker.private.json','official-control-services.private.json'):
    if (W/name).exists():collect(read(W/name))
assert secrets
seen=set();files={}
def scan(name,data,depth=0):
    digest=sha(data)
    if digest in seen:return
    seen.add(digest)
    assert not any(secret in data for secret in secrets),'Credential detected in '+name
    if data.startswith(b'PK\x03\x04'):
        assert depth<6
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert z.testzip() is None
            for item in z.infolist():
                if not item.is_dir():scan(name+'/'+item.filename,z.read(item),depth+1)
def add(name,data):
    scan(name,data);files[name]=data
for file in ROOT.rglob('*'):
    if not file.is_file():continue
    rel=file.relative_to(ROOT)
    if any(p=='__pycache__' or p.startswith('inspect-') for p in rel.parts):continue
    if file.suffix in ('.pyc','.pyo') or file.name.endswith(('-wal','-shm')) or file.name=='artifact-manifest.json':continue
    assert 'private' not in file.name.lower() and file.name!='.env'
    add('experiment/'+rel.as_posix(),file.read_bytes())
for name in ('run_v244_v245.py','report_v244_v245.py','package_v244_v245.py','run_v240_experiment.py','v242_capture_worker.py',
             'replay_v245_offline.py','finalize_v244_v245.py',
             'v244-tests.log','v245-tests.log','v244-live.log','v245-live.log'):
    add('runner/'+name,(W/name).read_bytes())
freeze=read(ROOT/'freeze.json')
for variant,commit in {**freeze['source_commits'],'v243':freeze['base_source']}.items():
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--','focused-rag','build_focused_submission.py'],cwd=REPO,text=True).splitlines()
    for name in names:add('source/'+variant+'/'+name,subprocess.check_output(['git','show',commit+':'+name],cwd=REPO))
code=W/'v230-benchmark-code/benchmarks'
for folder in ('platform_proxy','v8_deepseek/engine'):
    for file in (code/folder).rglob('*'):
        if file.is_file() and file.suffix in ('.py','.json','.md','.txt') and '__pycache__' not in file.parts:
            assert 'private' not in file.name.lower()
            add('benchmark-code/'+file.relative_to(code).as_posix(),file.read_bytes())
manifest={name:{'sha256':sha(data),'bytes':len(data),'object':'objects/'+sha(data)} for name,data in sorted(files.items())}
readme=b'''v244 / v245 frozen experiment audit

manifest.json maps logical paths to objects/<sha256>. Identical bytes are stored once.
Reconstruct only safe relative logical paths, verifying hashes; never write outside
your chosen destination. experiment/REPORT.html is the readable full comparison.
experiment/freeze.json fixes sources, packages, 30 known questions and v243 baseline.
All successful and failed attempts, caches and logs are preserved. source/ includes
frozen product code and tests. Private API configuration and BGE weights are absent.
The original runner retains Windows paths and is not a one-command portable install.

These are previously exposed regression questions, not an independent holdout.
50 means a tie with v243, not correctness or a predicted PSB score. The unchanged
two-order judge can contradict its own rationale. Raw results were not rerolled.
Runtimes use shared exact caches and are not cold platform latency comparisons.
v244 changes conditional retrieval; v245 changes only the final answer review.
The v243 platform reference 37.5/100 and 484 seconds is user-reported.
'''
add('README.txt',readme)
archive=OUT/'v244-v245-AUDIT.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.writestr('README.txt',readme);z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    stored=set()
    for name,row in manifest.items():
        if row['object'] not in stored:z.writestr(row['object'],files[name]);stored.add(row['object'])
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,row in manifest.items():assert sha(z.read(row['object']))==row['sha256']
result={'archive':archive.name,'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,
        'logical_files':len(manifest),'unique_objects':len(stored),'recursive_known_credential_scan':'passed',
        'packages':freeze['packages']}
(ROOT/'artifact-manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
