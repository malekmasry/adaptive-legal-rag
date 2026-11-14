"""Validate both V8 judge passes against predeclared manual sanity controls.

Does not call a model. Missing judgments mean NOT_RUN, not a passed calibration.
Stored scores are recomputed from validated proof IDs and raw model responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import v8_absolute_judge as judge

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / 'v8/calibration'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _parse_raw(value):
    if not isinstance(value, str):
        raise ValueError('Missing raw judge response')
    value = value.strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return json.loads(value)


def _diagnostics(judgment, expected):
    errors = []
    for key, value in expected.items():
        if key == 'extra_contradiction_penalty':
            ok = judgment[key] == value
        elif key == 'must_have_unverified_extra':
            ok = any(x['verdict'] == 'unverified' for x in judgment['extra_claims']) == value
        elif key == 'must_have_material_contradicted_extra':
            ok = any(x['verdict'] == 'contradicted' and x['material'] for x in judgment['extra_claims']) == value
        else:
            raise ValueError('Unknown calibration expectation: ' + key)
        if not ok:
            errors.append(key)
    return errors


def check(details=CONTROL / 'judgments.jsonl', *, output=CONTROL / 'check.json', control_dir=CONTROL,
          corpus=ROOT / 'knowledge-base.zip', services=None):
    details, control_dir, corpus = Path(details), Path(control_dir), Path(corpus)
    expected_path = control_dir / 'expected.json'
    expected = json.loads(expected_path.read_text())
    manifest = json.loads((control_dir / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if _sha(control_dir / name) != digest:
            raise ValueError('Calibration fixture changed: ' + name)
    if _sha(corpus) != manifest['corpus_sha256']:
        raise ValueError('Calibration corpus changed')
    base = {'purpose': 'Predeclared sanity checks, not an estimate of judge reliability or PSB agreement',
            'passed': False, 'control_count': len(expected), 'new_api_calls': 0,
            'expected_sha256': _sha(expected_path), 'manifest_sha256': _sha(control_dir / 'manifest.json'),
            'evaluator_sha256': _sha(ROOT / 'v8_absolute_judge.py')}
    if not details.exists():
        result = {**base, 'status': 'NOT_RUN', 'semantic_llm_calibration': 'NOT_RUN',
                  'checks': [], 'reason': 'No saved LLM judgments. Offline fixture validation is separate.'}
        if output is not None:
            _write(output, result)
        return result

    docs = judge.read_docs(corpus)
    cases = judge.load_cases(control_dir / 'cases.jsonl', docs)
    ids = [row['id'] for row in cases]
    if set(ids) != set(expected) or len(ids) != manifest['count']:
        raise ValueError('Calibration expectations/cases mismatch')
    answers_path = control_dir / 'answers.csv'
    answers = judge.read_answers(answers_path, ids)
    if set(answers) != set(ids):
        raise ValueError('Calibration answers incomplete')
    meta_path = details.with_suffix('.meta.json')
    meta = json.loads(meta_path.read_text())
    required_meta = {'protocol': judge.PROTOCOL, 'rubric': judge.RUBRIC, 'packing': judge.PACKING,
                     'model': judge.DEFAULT_MODEL, 'scope': 'calibration', 'ids': ids, 'passes': 2,
                     'execution_mode': 'live_api',
                     'prompt_sha256': hashlib.sha256(judge.PROMPT.encode()).hexdigest(),
                     'inputs': {'candidate': _sha(answers_path), 'cases': _sha(control_dir / 'cases.jsonl'),
                                'corpus': _sha(corpus)},
                     'implementation': {'v8_absolute_judge.py': _sha(ROOT / 'v8_absolute_judge.py')}}
    for key, value in required_meta.items():
        if meta.get(key) != value:
            raise ValueError('Calibration metadata differs: ' + key)
    if services is not None:
        config, _ = judge._configuration(services, judge.DEFAULT_MODEL)
        for name, value in [('judge_endpoint_sha256', config[0]), ('credential_fingerprint', config[1])]:
            if meta.get(name) != hashlib.sha256(value.encode()).hexdigest():
                raise ValueError('Calibration service identity differs: ' + name)
    if meta.get('resume_key') != judge.object_sha({k: v for k, v in meta.items() if k != 'resume_key'}):
        raise ValueError('Calibration resume identity differs')
    rows = [json.loads(line) for line in details.read_text().splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or not set(by_id) <= set(ids):
        raise ValueError('Duplicate or unrelated calibration row')
    checks = []
    for case in cases:
        key = case['id']
        e = expected[key]
        row = by_id.get(key)
        errors, scores = [], []
        if row is None:
            errors.append('missing_judgment_row')
        else:
            evaluation_key = judge.object_sha({'run': meta['resume_key'], 'case': case, 'answer': answers[key]})
            judge._validate_saved(row, case, answers[key], docs, evaluation_key,
                                  expected_model=judge.DEFAULT_MODEL, require_live=True)
            for i, one_pass in enumerate(row.get('rounds', [])):
                if one_pass.get('status') != 'ok':
                    errors.append(f'pass_{i+1}_invalid')
                    continue
                attempts = one_pass['attempts']
                if len(attempts) != 1 or attempts[0].get('http_status') != 200 or attempts[0].get('finish_reason') not in (None, 'stop'):
                    errors.append(f'pass_{i+1}_missing_successful_response')
                    continue
                payload = judge.make_payload(case, docs, answers[key], i)
                raw = judge.validate_round(_parse_raw(attempts[0].get('raw_content')), payload, docs)
                rescored = {**raw, **judge.score_round(raw)}
                if rescored != one_pass['judgment']:
                    raise ValueError('Stored control assessment differs from raw response: ' + key)
                scores.append(rescored['score'])
                if rescored['dataset_issue']:
                    errors.append(f'pass_{i+1}_dataset_issue')
                if rescored['score'] != e['score']:
                    errors.append(f'pass_{i+1}_score')
                errors.extend(f'pass_{i+1}_{problem}' for problem in _diagnostics(rescored, e['diagnostics']))
            if len(scores) != 2:
                errors.append('requires_two_valid_passes')
        checks.append({'id': key, 'kind': e['kind'], 'expected_score': e['score'],
                       'observed_pass_scores': scores, 'errors': errors, 'passed': not errors})
    passed = len(rows) == len(expected) and all(x['passed'] for x in checks)
    result = {**base, 'status': 'PASSED' if passed else ('INCOMPLETE' if len(rows) < len(expected) else 'FAILED'),
              'semantic_llm_calibration': 'PASSED' if passed else 'NOT_PASSED',
              'passed': passed, 'checks': checks, 'details_sha256': _sha(details), 'meta_sha256': _sha(meta_path),
              'saved_rows': len(rows), 'valid_control_count': sum(c['passed'] for c in checks),
              'current_service_identity_checked': services is not None}
    if output is not None:
        _write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details', type=Path, default=CONTROL / 'judgments.jsonl')
    parser.add_argument('--out', type=Path, default=CONTROL / 'check.json')
    parser.add_argument('--services', type=Path, help='Require the same endpoint and credentials as the calibrated judge')
    args = parser.parse_args()
    result = check(args.details, output=args.out, services=args.services)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(2 if result['status'] == 'NOT_RUN' else 1)
