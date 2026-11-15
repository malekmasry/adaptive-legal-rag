"""Copy completed evidence only; branch creation, commits and pushes are explicit."""
import argparse,hashlib,json,subprocess
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment';OUT=W.parent/'outputs'
REPO=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
SOURCES={
 'v244':('codex/v244-conditional','ed60eabc37b4897e6cf14c96f862360e3f818fc3'),
 'v245':('codex/v245-targeted-review','d88fd4d5440c2eb33f9c982f9d2d5ee402b4c8f8'),
 'benchmark':('codex/benchmark-v245','97a4b9b8d00ab5b3b432e6b3647bccdc126202a2')}
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
parser=argparse.ArgumentParser();parser.add_argument('variant',choices=SOURCES);args=parser.parse_args()
branch,source=SOURCES[args.variant]
assert git('branch','--show-current')==branch
assert git('rev-parse','HEAD')==source
assert not git('status','--porcelain')
assert read(ROOT/'manual-audit.json')['status']=='completed'
manifest=read(ROOT/'artifact-manifest.json')
assert manifest['recursive_known_credential_scan']=='passed'
assert sha((OUT/manifest['archive']).read_bytes())==manifest['sha256']
secrets=[]
def collect(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if isinstance(v,str) and len(v)>=12 and any(s in k.lower() for s in ('key','token','secret','authorization')):secrets.append(v.encode())
   elif isinstance(v,(dict,list)):collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
for name in ('platform-proxy-services.private.json','local_api.json','siliconflow-reranker.private.json','official-control-services.private.json'):
 if (W/name).exists():collect(read(W/name))
assert secrets
copied={}
def put(data,dest):
 assert not any(s in data for s in secrets),'Credential in '+str(dest)
 assert not dest.exists(),str(dest)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 copied[dest.relative_to(REPO).as_posix()]=sha(data)
target=REPO/'reports/v244-v245'
assert not target.exists()
put(b'* -text whitespace=cr-at-eol\n',target/'.gitattributes')
for file in sorted(ROOT.iterdir()):
 if file.is_file() and file.suffix in ('.json','.jsonl','.html','.txt','.log'):
  put(file.read_bytes(),target/file.name)
put((OUT/manifest['archive']).read_bytes(),target/manifest['archive'])
if args.variant=='benchmark':
 dest=REPO/'benchmarks/experiments/v244-v245'
 put(b'* -text whitespace=cr-at-eol\n',dest/'.gitattributes')
 for name in ('run_v244_v245.py','report_v244_v245.py','package_v244_v245.py','replay_v245_offline.py','finalize_v244_v245.py','publish_v244_v245.py'):
  put((W/name).read_bytes(),dest/name)
 note='''Completed v244/v245 regression experiment, 2026-11-15.
Read reports/v244-v245/report.txt before interpreting any metric.
The full audit has frozen sources/packages, unchanged judge code, raw two-order
judgments, all answers, captures, exact caches and logs. These 30 questions were
already exposed; they are not an independent holdout. The sole numeric v245 win
contradicts its written first-round rationale. Raw scores are retained unchanged.
Original runner scripts reference local Windows paths and private configurations;
they are archival, not a one-command portable install. run_v244_v245.py can spend
API tokens. Viewing reports or replay_v245_offline.py requires no model calls.
Parent: codex/benchmark-v243. Moonvolk benchmark ancestor: f19a327.
'''
 put(note.encode(),dest/'README.txt')
else:
 name=args.variant+'.zip';data=(ROOT/'packages'/name).read_bytes()
 assert sha(data)==manifest['packages'][name]['sha256']
 put(data,REPO/'submissions'/name)
publication={'branch':branch,'source_commit':source,'audit_sha256':manifest['sha256'],
 'files':copied,'credential_scan':'verified recursive archive scan plus matching hashes and plaintext scan'}
(OUT/(args.variant+'-publication.json')).write_text(json.dumps(publication,indent=2)+'\n',encoding='utf8')
print(json.dumps({'branch':branch,'files':len(copied),'audit_verified':True}))
