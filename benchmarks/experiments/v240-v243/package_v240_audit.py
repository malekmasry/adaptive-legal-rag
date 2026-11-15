"""Freeze all successful and failed experiment evidence into a secret-scanned ZIP."""
import argparse,hashlib,io,json,subprocess,zipfile
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['v240','v243'],default='v240');args=parser.parse_args()
W=Path(__file__).resolve().parent;ROOT=W/('v240-experiment' if args.variant=='v240' else 'v242-experiment');OUT=W.parent/'outputs'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
digest=lambda b:hashlib.sha256(b).hexdigest()
assert (ROOT/('holdout-summary.json' if args.variant=='v240' else 'v243-holdout-summary.json')).exists() and (ROOT/'selection.json').exists()
secrets=[]
def find_secrets(obj):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if isinstance(value,str) and len(value)>=12 and any(x in key.lower() for x in ('key','token','secret','authorization')):
                secrets.append(value.encode())
            elif isinstance(value,(dict,list)):find_secrets(value)
    elif isinstance(obj,list):
        for value in obj:find_secrets(value)
for name in ['platform-proxy-services.private.json','local_api.json','siliconflow-reranker.private.json']:
    if (W/name).exists():find_secrets(read(W/name))
assert secrets
seen=set()
def scan(name,data,depth=0):
    key=digest(data)
    if key in seen:return
    seen.add(key)
    assert not any(s in data for s in secrets),'Credential detected in '+name
    if data.startswith(b'PK\x03\x04'):
        assert depth<6
        with zipfile.ZipFile(io.BytesIO(data)) as nested:
            assert nested.testzip() is None,name
            for info in nested.infolist():
                if not info.is_dir():scan(name+'/'+info.filename,nested.read(info),depth+1)

files={}
def add(logical,file):
    data=Path(file).read_bytes();scan(logical,data);files[logical]=data
for file in ROOT.rglob('*'):
    if not file.is_file():continue
    rel=file.relative_to(ROOT)
    if any(part=='__pycache__' or part.startswith('inspect-') for part in rel.parts):continue
    if file.suffix in ('.pyc','.pyo') or file.name.endswith(('-wal','-shm')):continue
    if file.name in ('artifact-manifest.json',):continue
    assert 'private' not in file.name.lower() and file.name!='.env'
    add('experiment/'+rel.as_posix(),file)
for name in ['prepare_v240_experiment.py','normalize_v240_cases.py','freeze_v240_runtime.py',
    'run_v240_experiment.py','v240_capture_worker.py','resume_v240_experiment.py','inspect_v240_results.py',
    'report_v240_experiment.py','select_v240_experiment.py','package_v240_audit.py','v240-tests.log','v241-tests.log',
    'v241-tests-original-rule-expectation.log','v240-development.log','v240-development-resumed.log','v240-holdout.log',
    'run_v242_experiment.py','v242_capture_worker.py','evaluate_v243.py','report_v243.py','v242-tests.log','v243-tests.log',
    'v242-development.log','v243-development-replay.log','v243-holdout.log','finalize_v243_rendering.py','v243-final-tests.log']:
    if (W/name).exists():add('runner/'+name,W/name)
code=W/'v230-benchmark-code/benchmarks'
for directory in ['platform_proxy','v8_deepseek/engine']:
    for file in (code/directory).rglob('*'):
        if file.is_file() and file.suffix in ('.py','.json','.md','.txt') and '__pycache__' not in file.parts:
            assert 'private' not in file.name.lower()
            add('benchmark-code/'+file.relative_to(code).as_posix(),file)
repo=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
commits=read(ROOT/'runtime-freeze.json')['source_commits'] if args.variant=='v240' else {'v242':read(ROOT/'source-freeze.json')['source_commit'],'v243':read(ROOT/'v243-source-freeze.json')['source_commit']}
if args.variant=='v243':commits['v243-final']=read(ROOT/'final-rendering-verification.json')['source_commit']
for variant,commit in commits.items():
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,'--','focused-rag','build_focused_submission.py'],cwd=repo,text=True).splitlines()
    for name in names:
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        logical='source/'+variant+'/'+name;scan(logical,data);files[logical]=data
manifest={n:{'sha256':digest(b),'bytes':len(b),'object':'objects/'+digest(b)} for n,b in sorted(files.items())}
readme='''PSB diagnostic experiment audit

manifest.json maps each logical path to objects/<sha256>; identical bytes are stored once.
Reconstruct by writing each referenced object to its logical path. Do not extract paths
outside your target directory. All paths in this archive are relative and traversal-free.

experiment/REPORT.html: readable answers, goldens, two-order judge results, context and spend.
experiment/selection.json: choice frozen before holdout.
experiment/generated/: full run captures and original execution artifacts.
experiment/review-continuations/ (where present): exact missing reviews, original draft reused.
experiment/calls.jsonl and caches: all paid requests, failures and exact cache reuse.
experiment/packages/: immutable product submission ZIPs for the variants of this experiment.
runner/: authoring, freeze, observer, recovery, analysis, packaging and test logs.
benchmark-code/: the unchanged benchmark implementation used for this run.

These are local pairwise results against a fresh v239 baseline, not platform scores.
18 development +12 holdout authored before runs. Same corpus/author is a limitation.
Any source-only oracle is diagnostic and cannot be deployed. Successful completed
answers were not regenerated during balance recovery. Timings use shared caches.
The v242/v243 experiment adds all three requested question-brief features. Its five
development UI-label cleanups were replayed exactly, without new generation; the
final v243 was then executed live on holdout. Raw v242 scores are retained.
Private API configuration and BGE model weights are intentionally not distributed.
Runners retain original workspace paths; adaptation and private service config are
needed on another machine. No submission or claim of 80/85 platform points is made.
'''.encode()
scan('README.txt',readme)
archive=OUT/('v240-v241-AUDIT.zip' if args.variant=='v240' else 'v243-AUDIT.zip');OUT.mkdir(exist_ok=True)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.writestr('README.txt',readme);z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    stored=set()
    for name,data in files.items():
        key=manifest[name]['object']
        if key not in stored:z.writestr(key,data);stored.add(key)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,row in manifest.items():assert digest(z.read(row['object']))==row['sha256']
result={'archive':archive.name,'sha256':digest(archive.read_bytes()),'bytes':archive.stat().st_size,
    'logical_files':len(files),'unique_objects':len(stored),'recursive_known_credential_scan':'passed',
    'packages':read(ROOT/'analysis.json')['verified_package_sha256']}
(ROOT/'artifact-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
