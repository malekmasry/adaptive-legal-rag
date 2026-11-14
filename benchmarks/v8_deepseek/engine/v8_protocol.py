"""Freeze V8 inputs and preserve the exact three historical runtime archives."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from v8_absolute_judge import load_cases, read_docs, sha, write_json
from model_roles import JUDGE_MODEL

ROOT = Path(__file__).resolve().parent
V = ROOT / 'v8'
NAMES = ('conditional35', '35v8', 'rag_selected')
CODE = ('v8_protocol.py', 'v8_run.py', 'v8_compare.py', 'v8_absolute_judge.py',
        'v8_build_controls.py', 'v8_check_controls.py', 'v7_protocol.py', 'v8_security_run.py')


def verify():
    from v6_execute import implementation_hashes, verify_archive
    plan = json.loads((V / 'plan.json').read_text())
    if plan.get('judge_model') != JUDGE_MODEL:
        raise ValueError('DeepSeek judge is required for this benchmark version')
    frozen = json.loads((V / 'freeze.json').read_text())
    if sha(V / 'plan.json') != frozen['plan_sha256']:
        raise ValueError('V8 plan changed after freeze')
    if plan['implementations'] != implementation_hashes():
        raise ValueError('Frozen generation harness changed')
    for path, expected in frozen['inputs'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen V8 input changed: ' + path)
    for name, entry in plan['packages'].items():
        verify_archive(V / 'packages' / entry['package'], entry)
    return plan


def freeze():
    from v6_execute import implementation_hashes, verify_archive
    if (V / 'freeze.json').exists():
        return verify()
    readiness = json.loads((V / 'baselines.json').read_text())
    docs = read_docs(ROOT / 'knowledge-base.zip'); datasets = {}; clusters = {}
    source_owners = {}; family_owners = {}; contrast_owners = {}
    for name in ('development', 'holdout'):
        path = V / 'dataset' / (name + '.jsonl')
        rows = load_cases(path, docs)
        cluster_ids = {r.get('source_cluster', r.get('cluster_id')) for r in rows}
        if None in cluster_ids or '' in cluster_ids:
            raise ValueError('Each question needs a source cluster')
        for row in rows:
            cluster = row.get('source_cluster', row.get('cluster_id'))
            for source in {item['path'] for item in row['evidence']}:
                if source_owners.setdefault(source, (name, cluster)) != (name, cluster):
                    raise ValueError('Evidence document shared across source clusters or splits')
            family = row.get('family_id', row.get('family'))
            if family and family_owners.setdefault(family, name) != name:
                raise ValueError('Question family shared across splits')
            contrast = row.get('contrast_pair_id', row.get('contrast_group'))
            if contrast and contrast_owners.setdefault(contrast, (name, cluster)) != (name, cluster):
                raise ValueError('Contrast pair shared across clusters or splits')
        clusters[name] = cluster_ids
        datasets[name] = {'path': str(path.relative_to(V)), 'cases_sha256': sha(path), 'count': len(rows)}
    if clusters['development'] & clusters['holdout']:
        raise ValueError('Source clusters leak between splits')
    packages = {}
    (V / 'packages').mkdir(exist_ok=True)
    for name in NAMES:
        original = readiness['packages'][name]
        entry = {k: original[k] for k in ('package', 'sha256', 'files', 'family', 'role')}
        if 'commit' in original:
            entry['commit'] = original['commit']
        src = ROOT / original['source_archive']; dst = V / 'packages' / entry['package']
        verify_archive(src, entry)
        if dst.exists() and sha(dst) != entry['sha256']:
            raise ValueError('Existing V8 package differs')
        if not dst.exists():
            shutil.copyfile(src, dst)
        packages[name] = entry
    cache = V / 'embedding-cache-seed.sqlite3'
    if not cache.exists():
        shutil.copyfile(ROOT / 'v7/embedding-cache-checkpoint.sqlite3', cache)
    plan = {'protocol': 'v8-source-spans-and-contrast-cases',
            'workers': 3, 'outer_seconds_per_system': 1200,
            'generation_engine': 'unchanged v6_execute.py',
            'public_settings': readiness['services']['public_settings'],
            'service_profile_sha256': readiness['services']['service_profile_sha256'],
            'corpus_sha256': sha(ROOT / 'knowledge-base.zip'),
            'implementations': implementation_hashes(), 'datasets': datasets,
            'packages': packages, 'judge_model': JUDGE_MODEL, 'judge_passes': 2,
            'cache_seed_sha256': sha(cache),
            'cache_policy': 'Identical frozen SQLite seed copied separately per system and split; initialization included, BGE server warmup excluded.',
            'timing_limit': '1200 seconds per run; separate split timings do not prove a combined full submission meets its deadline.',
            'selection': 'No automatic winner or fixed +5 threshold; full development comparison with source-cluster uncertainty and saturation diagnostics.',
            'historical_psb_configuration_confirmed': False, 'organizer_equivalence': False,
            'holdout': 'New questions on historically exposed corpus; not pristine knowledge. Locked runtime packages before opening.',
            'paid_calls_at_freeze': 0}
    write_json(V / 'plan.json', plan)
    paths = [ROOT / p for p in CODE] + [ROOT / 'knowledge-base.zip', V / 'baselines.json', cache,
            ROOT / 'cache_seed/bge_embeddings.npz', ROOT / 'cache_seed/embedding_cache.json',
            V / 'services.example.json', V / 'requirements-runtime.txt']
    for folder in ('dataset', 'calibration', 'security'):
        paths.extend(p for p in (V / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.lock') and not p.name.startswith(('judgments', 'check.json', 'results')))
    frozen = {'at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': sha(V / 'plan.json'),
              'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
              'policy': 'No data, rubric or runtime edits after live calibration or generation; changes require a new version.'}
    write_json(V / 'freeze.json', frozen)
    verify()
    return plan


def open_holdout():
    plan = verify()
    record = {'freeze_sha256': sha(V / 'freeze.json'),
              'cases_sha256': plan['datasets']['holdout']['cases_sha256'],
              'packages': {n: e['sha256'] for n, e in plan['packages'].items()}}
    path = V / 'holdout-opened.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in record.items()):
            raise ValueError('Holdout identity changed')
    else:
        write_json(path, {**record, 'at': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--freeze', action='store_true')
    args = p.parse_args(); plan = freeze() if args.freeze else verify()
    print(json.dumps({'verified': True, 'systems': list(plan['packages']),
                      'counts': {n: d['count'] for n, d in plan['datasets'].items()}}, ensure_ascii=False))
