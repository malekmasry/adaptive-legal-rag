"""Freeze and verify V7 inputs without changing any V6 measurements.

Generation uses the existing frozen v6_execute engine. Evaluation is independent
per answer and therefore has a separate versioned identity.
"""
from __future__ import annotations

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')


import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

from v6_execute import implementation_hashes, verify_archive, verify_runtime

ROOT = Path(__file__).resolve().parent
V = ROOT / 'v7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def package_entry(path, **extra):
    path = Path(path)
    with ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate package member')
        files = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names}
    entry = {'package': path.name, 'sha256': sha(path), 'files': files,
             'family': 'selected', 'role': 'candidate', **extra}
    verify_archive(path, entry)
    return entry


def validate_generation(plan, dataset, name, directory):
    """Fail closed on partial runs, provenance drift, or modified emitted data."""
    from v6_execute import (assessment, observation_hashes, observe,
                            read_emitted_answers, profile_sha)
    directory = Path(directory)
    entry = plan['packages'][name]
    meta = json.loads((directory / 'run.meta.json').read_text())
    if meta.get('launch_status') != 'finished' or meta.get('exit_code') != 0:
        raise ValueError('Run is active or incomplete; no silent replacement')
    if (meta.get('system') != name or meta.get('package_entry') != entry
            or meta.get('dataset_entry') != plan['datasets'][dataset]
            or meta.get('cases_sha256') != plan['datasets'][dataset]['cases_sha256']
            or meta.get('package_sha256') != entry['sha256']
            or meta.get('corpus_sha256') != plan['corpus_sha256']
            or meta.get('service_profile_sha256') != plan['service_profile_sha256']
            or profile_sha(meta.get('public_settings', {})) != plan['service_profile_sha256']
            or meta.get('implementations') != plan['implementations']):
        raise ValueError('Generation provenance differs')
    verify_runtime(directory, entry['files'])
    for field, relative in [('answers_sha256', 'submission.csv'),
                            ('questions_sha256', 'data/questions.csv'),
                            ('corpus_sha256', 'data/kb.zip')]:
        if sha(directory / relative) != meta.get(field):
            raise ValueError('Generation artifact changed: ' + relative)
    if observation_hashes(directory) != meta.get('observation_sha256'):
        raise ValueError('Generation telemetry changed')
    answers, issues = read_emitted_answers(directory / 'submission.csv', meta['ids'])
    obs, usage, calls = observe(directory, entry['family'], meta['ids'], entry['files'])
    measured = assessment(meta, obs, answers, issues, meta['ids'])
    if (not measured['valid_run'] or not meta.get('valid_run')
            or obs != meta.get('service_observations')
            or usage != meta.get('token_usage') or calls != meta.get('api_calls_count')):
        raise ValueError('Generation failed measured validity checks')
    return meta


def verify_freeze():
    plan = json.loads((V / 'plan.json').read_text())
    freeze = json.loads((V / 'freeze.json').read_text())
    if sha(V / 'plan.json') != freeze['plan_sha256']:
        raise ValueError('Plan changed after freeze')
    if plan['implementations'] != implementation_hashes():
        raise ValueError('Frozen generation engine changed')
    for relative, expected in freeze['inputs'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    for name, entry in plan['packages'].items():
        verify_archive(V / 'packages' / entry['package'], entry)
    return plan, freeze


def record_holdout_open():
    plan, freeze = verify_freeze()
    path = V / 'holdout-opened.json'
    record = {'protocol': 'v7-heldout-frozen-before-opening',
              'freeze_sha256': sha(V / 'freeze.json'),
              'cases_sha256': plan['datasets']['holdout']['cases_sha256'],
              'package_sha256': {n: e['sha256'] for n, e in plan['packages'].items()}}
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old[k] for k in record} != record:
            raise ValueError('Holdout commitment changed')
    else:
        write_once(path, {**record, 'at': datetime.now(timezone.utc).isoformat(),
                          'policy': 'No runtime, rubric, or label edits after this point.'})

