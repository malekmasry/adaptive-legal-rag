"""Run isolated document-security pairs; factual diagnostics are not a RAG ranking.

No runtime input capture exists in the archived packages, so attack exposure and
security success remain unknown even when both legal answers receive 100.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import v8_absolute_judge as judge
from v8_protocol import ROOT, V, NAMES, verify
from v8_run import invoke
from v7_protocol import validate_generation
from v8_compare import load_result
from v8.security.evaluate_pairs import assess_pair

SIDES = ('clean', 'perturbed')


def prepare(plan, pair, side, *, persist=False):
    """Derive a deterministic v6-compatible plan without changing the main freeze."""
    if not re.fullmatch(r'SEC\d{2}', pair['id']) or side not in SIDES:
        raise ValueError('Invalid security pair/side')
    entry = pair[side]; suite = V / 'security'
    cases, corpus = suite / entry['cases'], suite / entry['corpus']
    for path, expected in ((cases, entry['cases_sha256']), (corpus, entry['corpus_sha256'])):
        if not path.resolve().is_relative_to(suite.resolve()) or judge.sha(path) != expected:
            raise ValueError('Security input identity differs')
    rows = judge.load_cases(cases, judge.read_docs(corpus))
    if len(rows) != 1 or rows[0]['id'] != pair['id'] or hashlib.sha256(rows[0]['question'].encode()).hexdigest() != pair['question_sha256']:
        raise ValueError('Security case identity differs')
    key = pair['id'] + '-' + side
    derived = {**plan, 'corpus_sha256': entry['corpus_sha256'],
               'datasets': {key: {'path': str(cases.relative_to(V)), 'cases_sha256': entry['cases_sha256'], 'count': 1}},
               'security_pair': pair['id'], 'security_side': side,
               'main_plan_sha256': judge.sha(V / 'plan.json'),
               'main_freeze_sha256': judge.sha(V / 'freeze.json'),
               'security_manifest_sha256': judge.sha(suite / 'manifest.json')}
    path = V / 'security-plans' / (key + '.json')
    if path.exists() and json.loads(path.read_text()) != derived:
        raise ValueError('Existing security plan differs; preserve prior experiment')
    if persist and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True); judge.write_json(path, derived)
    return key, derived, path, cases, corpus


def locations(key, name):
    return (V / 'security-runs' / key / name,
            V / 'security-judgments' / key / (name + '.jsonl'),
            V / 'security-cache-copies' / key / (name + '.sqlite3'))


def command(stage, job, name, a):
    key, plan, path, cases, corpus = job; directory, judgments, cache = locations(key, name)
    if stage == 'generate':
        return [sys.executable, str(ROOT / 'v6_execute.py'), '--plan', str(path), '--name', name,
                '--package', str(V / 'packages' / plan['packages'][name]['package']),
                '--cases', str(cases), '--kb', str(corpus), '--services', str(a.services),
                '--output', str(directory), '--seed-dir', str(ROOT / 'cache_seed'),
                '--cache-db', str(cache), '--weights', str(a.weights), '--seed', '2026111408']
    return [sys.executable, str(ROOT / 'v8_absolute_judge.py'), '--candidate', str(directory / 'submission.csv'),
            '--cases', str(cases), '--kb', str(corpus), '--services', str(a.services), '--out', str(judgments),
            '--scope', 'security', '--model', plan['judge_model'], '--workers', '2', '--passes', '2']


def validate_run(job, name):
    key, plan, path, _, _ = job; directory, _, _ = locations(key, name)
    meta = validate_generation(plan, key, name, directory)
    if (meta.get('plan_sha256_at_launch') != judge.sha(path) or meta.get('seed') != 2026111408
            or meta.get('embedding_seed_sha256') != judge.sha(ROOT / 'cache_seed/bge_embeddings.npz')
            or meta.get('embedding_seed_manifest_sha256') != judge.sha(ROOT / 'cache_seed/embedding_cache.json')):
        raise ValueError('Security generation plan/order/embedding seed differs')
    return meta


def report(plan, pairs, names, services):
    config, _ = judge._configuration(services, plan['judge_model'])
    expected = {'model': plan['judge_model'], 'passes': 2, 'scope': 'security',
                'judge_endpoint_sha256': hashlib.sha256(config[0].encode()).hexdigest(),
                'credential_fingerprint': hashlib.sha256(config[1].encode()).hexdigest()}
    result = {'schema': 'v8-security-matrix-1', 'main_freeze_sha256': judge.sha(V / 'freeze.json'),
              'systems': {}, 'security_success': None, 'exposure': 'unknown',
              'selection_decision': None, 'part_of_main_ranking': False,
              'limitation': 'Archived runtimes do not capture final generator input; no exposure claim can be made.'}
    for name in names:
        assessments = []
        for pair in pairs:
            found = []; sources = {}
            for side in SIDES:
                job = prepare(plan, pair, side); key, _, _, cases, corpus = job
                directory, judgments, _ = locations(key, name)
                if directory.exists():
                    validate_run(job, name)
                if not judgments.exists():
                    found.append(None); continue
                validate_run(job, name)
                meta, rows, _, _ = load_result(judgments, directory / 'submission.csv', cases, corpus)
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError('Security judge service/model identity differs')
                found.append(next((row for row in rows if row['id'] == pair['id']), None))
                sources[side] = {'judgments_sha256': judge.sha(judgments), 'answers_sha256': judge.sha(directory / 'submission.csv')}
            assessments.append({**assess_pair(pair, *found, capture=None,
                                             provenance_validated=all(row is not None for row in found)),
                                'artifacts': sources})
        result['systems'][name] = assessments
    return result


def main(a):
    plan = verify(); manifest = json.loads((V / 'security/manifest.json').read_text())
    pairs = [p for p in manifest['pairs'] if not a.pairs or p['id'] in a.pairs]
    names = list(a.names)
    if not pairs or len(names) != len(set(names)) or not set(names) <= set(NAMES) or (a.pairs and set(a.pairs) != {p['id'] for p in pairs}):
        raise ValueError('Unknown or duplicate security selection')
    if a.stage in ('generate', 'all') and a.weights is None:
        raise ValueError('--weights required for generation')
    from v6_execute import public_settings, profile_sha
    config = json.loads(a.services.read_text())
    if any(not config.get(k) for k in ('LLM_API_KEY', 'LLM_BASE_URL', 'RERANKER_API_KEY', 'RERANKER_BASE_URL')) or profile_sha(public_settings(config)) != plan['service_profile_sha256']:
        raise ValueError('Service profile differs from main freeze')
    seed = json.loads((ROOT / 'cache_seed/embedding_cache.json').read_text())
    if judge.sha(ROOT / 'cache_seed/bge_embeddings.npz') != seed['cache_sha256']:
        raise ValueError('Embedding seed corrupted')
    if a.stage in ('generate', 'all'):
        for entry in json.loads((ROOT / 'bge_weights.json').read_text())['files']:
            if judge.sha(a.weights / entry['file']) != entry['sha256']:
                raise ValueError('Local BGE weights differ')
    jobs = [prepare(plan, pair, side) for pair in pairs for side in SIDES]
    for job in jobs:
        for name in names:
            directory, _, cache = locations(job[0], name)
            if directory.exists():
                validate_run(job, name)
            elif cache.exists() and judge.sha(cache) != plan['cache_seed_sha256']:
                raise ValueError('Unattributed modified security cache')
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'pairs': len(pairs), 'systems': names, 'variant_runs': len(jobs) * len(names),
                          'expected_answers': len(jobs) * len(names), 'planned_judge_passes': 2 * len(jobs) * len(names) if a.stage in ('all', 'judge') else 0,
                          'api_calls': 0, 'ports_opened': 0, 'exposure': 'unknown', 'security_success': None,
                          'semantic_calibration_required': True,
                          'example_commands': [command(s, jobs[0], names[0], a) for s in ('generate', 'judge')]}, ensure_ascii=False, indent=2))
        return
    from v8_check_controls import check
    if not check(services=a.services, output=None)['passed']:
        raise ValueError('Semantic LLM calibration has not passed for these services')
    with (V / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [prepare(plan, pair, side, persist=True) for pair in pairs for side in SIDES]
        for job in jobs:
            for name in names:
                directory, _, cache = locations(job[0], name)
                if a.stage in ('generate', 'all') and not directory.exists():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    if not cache.exists():
                        shutil.copyfile(V / 'embedding-cache-seed.sqlite3', cache)
                    invoke(command('generate', job, name, a), V / 'security-logs' / job[0] / ('generate-' + name + '.log'))
                    validate_run(job, name)
                if a.stage in ('judge', 'all'):
                    validate_run(job, name)
                    invoke(command('judge', job, name, a), V / 'security-logs' / job[0] / ('judge-' + name + '.log'))
        if a.stage in ('all', 'compare'):
            result = report(plan, pairs, names, a.services)
            target = V / 'security-reports' / ('matrix-' + '-'.join(p['id'] for p in pairs) + '-' + '-'.join(names) + '.json')
            target.parent.mkdir(parents=True, exist_ok=True); judge.write_json(target, result)
            print(json.dumps({'report': str(target), 'security_success': None, 'selection_decision': None}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['generate', 'judge', 'compare', 'all'], default='all')
    p.add_argument('--pairs', nargs='+'); p.add_argument('--names', nargs='+', default=list(NAMES))
    p.add_argument('--services', type=Path, required=True); p.add_argument('--weights', type=Path)
    p.add_argument('--dry-run', action='store_true'); main(p.parse_args())
