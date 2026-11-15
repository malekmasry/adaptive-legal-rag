import hashlib,json,py_compile,shutil,subprocess,zipfile
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v240-experiment'
REPO=Path(r'C:\Users\USRBGDY\.codex\worktrees\psb-retriever35-v8\adaptive-legal-rag')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'runtime-freeze.json').exists() and not (ROOT/'calls.jsonl').exists()
shutil.copyfile(W/'v239-final.zip',ROOT/'packages/v239.zip')
shutil.copyfile(W/'v230-comparison/embedding-cache.sqlite3',ROOT/'embedding-cache.sqlite3')
with zipfile.ZipFile(ROOT/'packages/v239.zip') as base:
    with zipfile.ZipFile(ROOT/'packages/v240.zip') as candidate:
        assert set(candidate.namelist())-set(base.namelist())=={'context_packing.py'}
        changes=[n for n in base.namelist() if base.read(n)!=candidate.read(n)]
        assert changes==['focused_context.py'],changes
    with zipfile.ZipFile(ROOT/'packages/v241.zip') as candidate:
        assert set(candidate.namelist())==set(base.namelist())
        changes=[n for n in base.namelist() if base.read(n)!=candidate.read(n)]
        assert changes==['pipeline.py'],changes
        code=candidate.read('pipeline.py').decode().replace('\r\n','\n')
        expected=base.read('pipeline.py').decode().replace('\r\n','\n').replace('        prepared=build_rule_draft(query,evidence,self.docs,self.s.as_of)',
            '        # Independent ablation: keep source hints and calculations, but let the\n        # generator formulate every answer before the existing review stage.\n        prepared=None')
        assert code==expected
for n in ['run_v240_experiment.py','v240_capture_worker.py']:py_compile.compile(str(W/n),doraise=True)
previous=read(W/'v230-comparison/freeze.json')
for n,h in previous['code_hashes'].items():assert sha(W/'v230-benchmark-code/benchmarks/platform_proxy'/n)==h
freeze={'dataset_freeze_sha256':sha(ROOT/'dataset-freeze.json'),
    'runner_sha256':sha(W/'run_v240_experiment.py'),'worker_sha256':sha(W/'v240_capture_worker.py'),
    'code_hashes':previous['code_hashes'],'embedding_code_hashes':previous['embedding_code_hashes'],'judge_prompt_sha256':previous['judge_prompt_sha256'],
    'packages':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (ROOT/'packages').glob('*.zip')},
    'source_commits':{n:subprocess.check_output(['git','rev-parse',branch],cwd=REPO,text=True).strip() for n,branch in [('v239','codex/v239'),('v240','codex/v240-context'),('v241','codex/v241-no-ready')]},
    'proof':'v240 changes only context assembly and adds its allocator; v241 changes only the one ready-draft assignment. Retrieval, prompts, hints, review, models and renderer are unchanged. External worker observes normal execution and separately supplies a marked source-only oracle.',
    'full_run_as_of':'2026-11-15','selection_and_holdout':'No paid model calls before this freeze. Product code will not be changed after results.'}
(ROOT/'runtime-freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(freeze['packages']),flush=True)
