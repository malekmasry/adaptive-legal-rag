"""Run calibration first, then the same frozen benchmark for all three systems.

Dry-run performs no networking, model calls, port binding or result mutation.
An interrupted generation directory is retained and must not silently be reused.
Judging resumes only under identical source, answer, configuration and code hashes.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from v8_absolute_judge import sha, write_json
from v8_protocol import ROOT, V, NAMES, verify, open_holdout


def validated_generation(plan, dataset, name):
    from v7_protocol import validate_generation
    meta = validate_generation(plan, dataset, name, V / 'runs' / dataset / name)
    if (meta.get('plan_sha256_at_launch') != sha(V / 'plan.json') or meta.get('seed') != 2026111408
            or meta.get('embedding_seed_sha256') != sha(ROOT / 'cache_seed/bge_embeddings.npz')
            or meta.get('embedding_seed_manifest_sha256') != sha(ROOT / 'cache_seed/embedding_cache.json')):
        raise ValueError('Generation plan, ordering seed or embedding seed differs')
    return meta


def comparison(dataset, plan, services):
    from v8_compare import compare
    from v8_absolute_judge import _configuration
    config, _ = _configuration(services, plan['judge_model'])
    expected = {'scope': dataset, 'model': plan['judge_model'], 'passes': 2,
                'judge_endpoint_sha256': hashlib.sha256(config[0].encode()).hexdigest(),
                'credential_fingerprint': hashlib.sha256(config[1].encode()).hexdigest()}
    for name in NAMES:
        meta = json.loads((V / 'judgments' / dataset / (name + '.meta.json')).read_text())
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError('Judged service, model or split differs from calibrated comparison')
    measured = {name: validated_generation(plan, dataset, name) for name in NAMES}
    result = compare({n: V / 'judgments' / dataset / (n + '.jsonl') for n in NAMES},
                   {n: V / 'runs' / dataset / n / 'submission.csv' for n in NAMES},
                   V / plan['datasets'][dataset]['path'], ROOT / 'knowledge-base.zip')
    result['runtime'] = {name: {key: meta.get(key) for key in ('elapsed_seconds', 'embedding_warmup_seconds',
                            'token_usage', 'api_calls_count', 'valid_run', 'outer_timeout', 'cache_policy')}
                         for name, meta in measured.items()}
    result['timing_limit_note'] = plan['timing_limit']
    return result


def command(stage, name, a, plan):
    cases = V / plan['datasets'][a.dataset]['path']; directory = V / 'runs' / a.dataset / name
    if stage == 'generate':
        return [sys.executable, str(ROOT / 'v6_execute.py'), '--plan', str(V / 'plan.json'),
                '--name', name, '--package', str(V / 'packages' / plan['packages'][name]['package']),
                '--cases', str(cases), '--kb', str(ROOT / 'knowledge-base.zip'),
                '--services', str(a.services), '--output', str(directory),
                '--seed-dir', str(ROOT / 'cache_seed'), '--cache-db', str(V / 'cache-copies' / a.dataset / (name + '.sqlite3')),
                '--weights', str(a.weights), '--seed', '2026111408']
    return [sys.executable, str(ROOT / 'v8_absolute_judge.py'), '--candidate', str(directory / 'submission.csv'),
            '--cases', str(cases), '--kb', str(ROOT / 'knowledge-base.zip'), '--services', str(a.services),
            '--out', str(V / 'judgments' / a.dataset / (name + '.jsonl')),
            '--scope', a.dataset, '--model', plan['judge_model'], '--workers', '2', '--passes', '2']


def calibration_commands(a, plan):
    return [[sys.executable, str(ROOT / 'v8_absolute_judge.py'), '--candidate', str(V / 'calibration/answers.csv'),
             '--cases', str(V / 'calibration/cases.jsonl'), '--kb', str(ROOT / 'knowledge-base.zip'),
             '--services', str(a.services), '--out', str(V / 'calibration/judgments.jsonl'),
             '--scope', 'calibration', '--model', plan['judge_model'], '--workers', '2', '--passes', '2'],
            [sys.executable, str(ROOT / 'v8_check_controls.py'), '--services', str(a.services)]]


def invoke(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        done = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError('Stage stopped; saved output retained. See ' + str(log))


def main(a):
    from model_roles import validate_roles
    validate_roles(json.loads(a.services.read_text()))
    plan = verify()
    from v8_absolute_judge import _configuration
    _configuration(a.services, plan['judge_model'])
    if a.stage in ('generate', 'all'):
        from v6_execute import profile_sha, public_settings
        if profile_sha(public_settings(json.loads(a.services.read_text()))) != plan['service_profile_sha256']:
            raise ValueError('Generation service profile differs from freeze')
    if a.stage in ('generate', 'all') and a.weights is None:
        raise ValueError('--weights is required for generation')
    if a.stage in ('generate', 'all'):
        for item in json.loads((ROOT / 'bge_weights.json').read_text())['files']:
            if sha(a.weights / item['file']) != item['sha256']:
                raise ValueError('Local BGE weights differ from the pinned encoder')
    if a.stage == 'calibrate':
        commands = calibration_commands(a, plan)
    else:
        stages = ('generate', 'judge') if a.stage == 'all' else (() if a.stage == 'compare' else (a.stage,))
        commands = [command(stage, name, a, plan) for name in NAMES for stage in stages]
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'generation_calls': 0, 'judge_calls': 0,
                          'commands': commands, 'comparison_requested': a.stage in ('all', 'compare'),
                          'calibration_required_before_generation': True,
                          'live_status': 'Not executed. DeepSeek judge calibration is required; old Qwen results cannot be reused.'}, ensure_ascii=False, indent=2))
        return
    with (V / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'calibrate':
            for i, cmd in enumerate(commands):
                invoke(cmd, V / 'logs' / ('calibration-' + str(i) + '.log'))
            return
        # Revalidate controls from their raw persisted rounds, never trust a hand-edited passed flag.
        invoke([sys.executable, str(ROOT / 'v8_check_controls.py'), '--services', str(a.services)], V / 'logs/calibration-check.log')
        if a.dataset == 'holdout':
            dev = comparison('development', plan, a.services)
            if not dev['complete'] or not dev['all_scored']:
                raise ValueError('Finish and validate development judging before opening holdout')
            open_holdout()
        for name in NAMES:
            directory = V / 'runs' / a.dataset / name
            if a.stage in ('generate', 'all'):
                if directory.exists():
                    validated_generation(plan, a.dataset, name)
                else:
                    cache = V / 'cache-copies' / a.dataset / (name + '.sqlite3')
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    if cache.exists() and sha(cache) != plan['cache_seed_sha256']:
                        raise ValueError('Unattributed modified embedding cache; will not reuse')
                    if not cache.exists():
                        shutil.copyfile(V / 'embedding-cache-seed.sqlite3', cache)
                    invoke(command('generate', name, a, plan), V / 'logs' / a.dataset / ('generation-' + name + '.log'))
                    validated_generation(plan, a.dataset, name)
            if a.stage in ('judge', 'all'):
                validated_generation(plan, a.dataset, name)
                invoke(command('judge', name, a, plan), V / 'logs' / a.dataset / ('judge-' + name + '.log'))
        if a.stage in ('all', 'compare'):
            result = comparison(a.dataset, plan, a.services)
            output = V / 'reports' / a.dataset / 'comparison.json'
            output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
            print(json.dumps({'complete': result['complete'], 'all_scored': result['all_scored'],
                              'selection_decision': None, 'health': result['health']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['calibrate', 'generate', 'judge', 'compare', 'all'], default='all')
    p.add_argument('--dataset', choices=['development', 'holdout'], default='development')
    p.add_argument('--services', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--dry-run', action='store_true')
    main(p.parse_args())
