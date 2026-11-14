"""Judge exact emitted CSV text with the unchanged two-order csv_judge evaluator.

Default: require completed valid generation. --allow-incomplete explicitly judges
nonempty emitted answers even after technical/deadline failures; missing answers
remain unknown, generation validity stays false, and ranking/selection is barred.
--reuse-judgments imports exact trusted v5 parent judgments without model calls.
Persisted needs_review/judge_error rows are never silently retried on resume.
"""
from __future__ import annotations

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import csv_judge
import v5_common_judge as legacy
from v4_compare import compare_systems, summarize_scores
from v6_execute import (ROOT, assessment, dataset_entry, implementation_hashes, load_plan,
    observe, observation_hashes, profile_sha, read_emitted_answers, verify_archive, verify_runtime)

PROTOCOL = 'fixed-bm25-common-reference-v6'
DEFAULT_MODEL = legacy.DEFAULT_MODEL
EVALUATE_SOURCE_SHA256 = legacy.EVALUATE_SOURCE_SHA256
sha, object_sha, write_json, load_cases = legacy.sha, legacy.object_sha, legacy.write_json, legacy.load_cases


def _trusted_v5(plan, root):
    path = root / 'v5/plan.json'
    if sha(path) != plan.get('trusted_v5', {}).get('plan_sha256'):
        raise ValueError('Untrusted or changed v5 registry')
    source = json.loads(path.read_text())
    for name, expected in source['implementations'].items():
        if sha(root / name) != expected:
            raise ValueError('Trusted v5 implementation source changed')
    return source


def verify_run(run_dir, rows, cases_sha256, corpus_sha256, *, plan, reference=False,
               allow_incomplete=False, root=ROOT):
    directory = Path(run_dir)
    meta_path = directory / 'run.meta.json'
    meta = json.loads(meta_path.read_text())
    is_v6 = meta.get('protocol') == 'frozen-runtime-v6'
    if is_v6:
        entry = plan['packages'].get(meta.get('system'))
        if entry is None or entry != meta.get('package_entry') or object_sha(entry) != meta.get('package_entry_sha256'):
            raise ValueError('Generation package entry changed')
        if meta.get('implementations') != implementation_hashes(root) or meta.get('runner_sha256') != sha(root / 'v6_execute.py'):
            raise ValueError('Generation implementation source mismatch')
        if plan.get('datasets', {}).get(meta.get('dataset')) != meta.get('dataset_entry') or meta['dataset_entry'].get('cases_sha256') != meta.get('cases_sha256'):
            raise ValueError('Generation cases absent from frozen plan')
        package = root / 'v6/packages' / entry['package']
        origin = 'v6_regenerated'
    else:
        trusted = _trusted_v5(plan, root)
        entry = trusted['packages'].get(meta.get('system'))
        if entry is None or meta.get('runner_sha256') != trusted['implementations']['v5_execute.py']:
            raise ValueError('Generation is not from trusted v5 runner')
        if meta.get('cases_sha256') != trusted['cases_sha256'] or trusted['corpus_sha256'] != corpus_sha256:
            raise ValueError('Original v5 cases/corpus mismatch')
        entry = {**entry, 'family': 'v8' if meta['system'] == '35v8' else 'selected'}
        package = root / 'v5/packages' / entry['package']
        origin = 'trusted_v5_generation_reused'
    if meta.get('package_sha256') != entry['sha256']:
        raise ValueError('Generation package fingerprint mismatch')
    verify_archive(package, entry)
    verify_runtime(directory, entry['files'])
    if reference and (meta.get('evaluation_role') != 'fixed_bm25_reference' or
                      (not is_v6 and meta.get('system') != 'reference') or
                      (is_v6 and entry.get('role') != 'fixed_bm25_reference')):
        raise ValueError('Reference does not have the frozen BM25 role')
    for field, path in (('questions_sha256', directory / 'data/questions.csv'),
                        ('corpus_sha256', directory / 'data/kb.zip')):
        if meta.get(field) != sha(path):
            raise ValueError('Generation artifact hash mismatch: ' + field)
    if meta.get('corpus_sha256') != corpus_sha256 or corpus_sha256 != plan.get('corpus_sha256'):
        raise ValueError('Generation/judging corpus differs')
    questions = legacy._questions(directory / 'data/questions.csv')
    if list(questions) != meta.get('ids'):
        raise ValueError('Generation input order/IDs differ from metadata')
    if any(questions.get(row['id']) != row['question'] for row in rows):
        raise ValueError('Selected ID/question differs from generation input')
    if meta.get('cases_sha256') == cases_sha256 and set(questions) != {r['id'] for r in rows}:
        raise ValueError('Exact cases hash with inconsistent IDs')
    settings = meta.get('public_settings')
    if not isinstance(settings, dict) or profile_sha(settings) != meta.get('service_profile_sha256') or meta.get('service_profile_sha256') != plan.get('service_profile_sha256'):
        raise ValueError('Generation service profile mismatch')
    csv_path = directory / 'submission.csv'
    if (csv_path.exists() and meta.get('answers_sha256') != sha(csv_path)) or (not csv_path.exists() and meta.get('answers_sha256') is not None):
        raise ValueError('Answer CSV fingerprint mismatch')
    answers, issues = read_emitted_answers(csv_path, list(questions))
    obs, usage, call_count = observe(directory, entry['family'], list(questions), entry['files'])
    actual = assessment(meta, obs, answers, issues, list(questions))
    hashes = observation_hashes(directory)
    if is_v6:
        if hashes != meta.get('observation_sha256') or obs != meta.get('service_observations'):
            raise ValueError('Generation telemetry changed')
        if any(meta.get(k) != actual[k] for k in ('valid_run', 'all_ids_valid', 'invalid_reasons', 'answer_csv_issues')):
            raise ValueError('Generation validity does not match observed execution')
        if usage != meta.get('token_usage') or call_count != meta.get('api_calls_count'):
            raise ValueError('Generation API accounting mismatch')
    else:
        # Old runs lack telemetry hashes: bind their exact log hashes now, and
        # independently validate observations rather than borrowing another run.
        previous = meta.get('service_observations', {})
        for key in ('dense_index_ready', 'rerank_success', 'generation_success', 'technical_errors', 'completed_answers'):
            if previous.get(key) != obs.get(key):
                raise ValueError('Trusted v5 observations do not match its own logs')
        if meta.get('valid_run') is not True or not actual['valid_run']:
            raise ValueError('Only complete trusted v5 generation can be reused')
    if not actual['valid_run'] and not allow_incomplete:
        raise ValueError('Incomplete generation requires explicit --allow-incomplete')
    provenance = {'origin': origin, 'package_sha256': entry['sha256'], 'runtime_file_sha256': entry['files'],
        'run_meta_sha256': sha(meta_path), 'answers_sha256': meta.get('answers_sha256'),
        'questions_sha256': meta['questions_sha256'], 'generation_cases_sha256': meta['cases_sha256'],
        'service_profile_sha256': meta['service_profile_sha256'], 'generation_ids': meta['ids'],
        'selected_ids': [r['id'] for r in rows], 'reused_subset': meta['cases_sha256'] != cases_sha256,
        'generation_valid': actual['valid_run'], 'generation_invalid_reasons': actual['invalid_reasons'],
        'observation_sha256': hashes, 'answer_statuses': obs['answer_statuses'], 'token_usage': usage,
        'generation_api_calls_count': call_count}
    return answers, provenance


def _cached_results(path, source_cases, rows, reference_run, candidate_run, reference_meta, candidate_meta, config, plan, root):
    """Only exact v5 evaluator outputs may be copied. No new model calls."""
    _trusted_v5(plan, root)
    path = Path(path)
    meta_path = path.with_suffix('.meta.json')
    old = json.loads(meta_path.read_text())
    source_rows = load_cases(source_cases)
    source_by_id = {r['id']: r for r in source_rows}
    if any(source_by_id.get(row['id']) != row for row in rows):
        raise ValueError('Cached judgment case definition differs')
    if old.get('protocol') != legacy.PROTOCOL or old.get('semantic_protocol') != csv_judge.PROTOCOL or old.get('model') != config[2]:
        raise ValueError('Cached evaluator protocol/model differs')
    if old.get('inputs') != {'cases_sha256': sha(source_cases), 'corpus_sha256': plan['corpus_sha256']} or old.get('ids') != [r['id'] for r in source_rows]:
        raise ValueError('Cached judge cases/corpus provenance differs')
    if old.get('prompt_sha256') != hashlib.sha256(csv_judge.PROMPT.encode()).hexdigest() or old.get('evaluate_source_sha256') != EVALUATE_SOURCE_SHA256 or old.get('judge_endpoint_sha256') != hashlib.sha256(config[0].encode()).hexdigest():
        raise ValueError('Cached semantic evaluator identity differs')
    if not old.get('implementation') or any(sha(root / name) != value for name, value in old['implementation'].items()):
        raise ValueError('Cached judge source hashes differ')
    if old.get('resume_key') != object_sha({k: v for k, v in old.items() if k != 'resume_key'}):
        raise ValueError('Cached judge metadata fingerprint differs')
    source_answers = []
    for role, directory, current in (('reference', reference_run, reference_meta), ('candidate', candidate_run, candidate_meta)):
        if current['origin'] != 'trusted_v5_generation_reused':
            raise ValueError('Cached parent judgments cannot label regenerated candidates')
        answers, verified = legacy.verify_run(directory, source_rows, sha(source_cases), plan['corpus_sha256'], reference=role == 'reference', root=root)
        if old.get(role) != verified or verified['run_meta_sha256'] != current['run_meta_sha256']:
            raise ValueError('Cached judgment run/source provenance differs')
        source_answers.append(answers)
    results = csv_judge.load_rows(path)
    by_id = {}
    for result in results:
        key = result['id']
        if key not in source_by_id:
            raise ValueError('Unrelated cached result ID')
        expected = object_sha({'run': old['resume_key'], 'case': source_by_id[key],
            'reference': source_answers[0][key], 'candidate': source_answers[1][key]})
        legacy._validate_result(result, source_by_id[key], expected)
        by_id[key] = result
    if any(row['id'] not in by_id for row in rows):
        raise ValueError('Requested cached judgments are incomplete; no automatic retry')
    summary_path = path.with_suffix('.summary.json')
    if not summary_path.exists() or json.loads(summary_path.read_text()).get('details_sha256') != sha(path):
        raise ValueError('Cached judgment details hash differs')
    return {row['id']: by_id[row['id']] for row in rows}, {
        'origin': 'trusted_v5_judgments_reused', 'details_sha256': sha(path), 'meta_sha256': sha(meta_path),
        'summary_sha256': sha(summary_path), 'source_resume_key': old['resume_key'],
        'source_cases_sha256': sha(source_cases), 'selected_ids': [r['id'] for r in rows]}


def _validate(result, row, key):
    if result.get('status') == 'generation_missing':
        if result.get('evaluation_key') != key or result.get('id') != row['id'] or result.get('score') is not None or result.get('rounds') != [] or result.get('attempts') != []:
            raise ValueError('Invalid missing-generation judgment')
    else:
        legacy._validate_result(result, row, key)


def run(cases, kb, reference_run, candidate_run, services, output, plan, workers=2,
        model=DEFAULT_MODEL, allow_incomplete=False, reuse_judgments=None, reuse_cases=None, root=ROOT):
    plan = load_plan(plan, root=root)
    if workers != plan.get('judge_workers', 2) or not 1 <= workers <= 4 or model != plan.get('judge', DEFAULT_MODEL):
        raise ValueError('Judge workers/model differ from frozen plan')
    rows = load_cases(cases)
    dataset_name, _ = dataset_entry(plan, cases, rows, kb)
    cases_hash, corpus_hash = sha(cases), sha(kb)
    reference, reference_meta = verify_run(reference_run, rows, cases_hash, corpus_hash, plan=plan, reference=True, root=root)
    candidate, candidate_meta = verify_run(candidate_run, rows, cases_hash, corpus_hash, plan=plan, allow_incomplete=allow_incomplete, root=root)
    config, secrets = legacy._configuration(services, model)
    if hashlib.sha256(config[0].encode()).hexdigest() != json.loads((Path(candidate_run) / 'run.meta.json').read_text())['public_settings']['endpoint_fingerprints']['LLM_BASE_URL']:
        raise ValueError('Judge endpoint differs from frozen service profile')
    cached, cached_meta = {}, None
    if reuse_judgments:
        cached, cached_meta = _cached_results(reuse_judgments, reuse_cases or root / 'v5/cases.jsonl', rows,
            reference_run, candidate_run, reference_meta, candidate_meta, config, plan, root)
    with ZipFile(kb) as archive:
        names = [name for name in archive.namelist() if name.endswith('.txt')]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate corpus document paths')
        docs = {name: archive.read(name).decode('utf-8-sig') for name in names}
    meta = {'protocol': PROTOCOL, 'semantic_protocol': csv_judge.PROTOCOL, 'scope': 'common_reference',
        'dataset': dataset_name, 'ids': [r['id'] for r in rows], 'model': model, 'workers': workers,
        'judge_endpoint_sha256': hashlib.sha256(config[0].encode()).hexdigest(),
        'prompt_sha256': hashlib.sha256(csv_judge.PROMPT.encode()).hexdigest(),
        'evaluate_source_sha256': EVALUATE_SOURCE_SHA256,
        'inputs': {'cases_sha256': cases_hash, 'corpus_sha256': corpus_hash},
        'reference': reference_meta, 'candidate': candidate_meta, 'implementation': implementation_hashes(root),
        'answer_orders': [['reference', 'candidate'], ['candidate', 'reference']],
        'allow_incomplete': allow_incomplete, 'cached_judgments': cached_meta,
        'generation_failure_policy': 'Judge exact nonempty emitted CSV texts; absent/empty/oversized rows unknown. Invalid generation is excluded from ranking and selection.',
        'organizer_equivalence': False, 'winner_selection': False,
        'attempt_retention': 'All attempts exposed by unchanged csv_judge.evaluate; no transport bodies or pre-logger rejected outputs'}
    run_key = object_sha(meta)
    meta['resume_key'] = run_key
    keys = {row['id']: object_sha({'run': run_key, 'case': row, 'reference': reference.get(row['id']),
        'candidate': candidate.get(row['id'])}) for row in rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected = {Path(p).resolve() for p in (cases, kb, services)} | {
        (Path(reference_run) / 'submission.csv').resolve(), (Path(candidate_run) / 'submission.csv').resolve()}
    if reuse_judgments:
        protected |= {Path(reuse_judgments).resolve(), Path(reuse_judgments).with_suffix('.meta.json').resolve(), Path(reuse_judgments).with_suffix('.summary.json').resolve()}
    destinations = [output, output.with_suffix('.meta.json'), output.with_suffix('.summary.json'), output.with_suffix('.lock')]
    if len(set(destinations)) != 4 or any(p.resolve() in protected for p in destinations):
        raise ValueError('Judgment outputs overlap inputs or one another')
    with output.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Judgment output already running') from None
        meta_path = output.with_suffix('.meta.json')
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != meta:
                raise ValueError('Cannot resume changed cases, sources, answers, profile, policy, or evaluator')
        elif output.exists():
            raise ValueError('Judgments exist without provenance')
        else:
            write_json(meta_path, meta)
        results = csv_judge.load_rows(output) if output.exists() else []
        by_id = {row['id']: row for row in rows}
        for result in results:
            if result['id'] not in by_id:
                raise ValueError('Unexpected resumed result ID')
            _validate(result, by_id[result['id']], keys[result['id']])
        done = {r['id'] for r in results}
        with output.open('a', encoding='utf-8', buffering=1) as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            def persist(result, row):
                key = row['id']
                result.update(generation_status=candidate_meta['answer_statuses'].get(key, 'missing'),
                    generation_run_valid=candidate_meta['generation_valid'])
                _validate(result, row, keys[key])
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                results.append(result)
                print(json.dumps({k: result.get(k) for k in ('id', 'status', 'score', 'judgment_origin')}), flush=True)
            for row in rows:
                key = row['id']
                if key in done:
                    continue
                if key in cached:
                    result = {**cached[key], 'source_evaluation_key': cached[key]['evaluation_key'],
                        'evaluation_key': keys[key], 'judgment_origin': 'trusted_v5_cache'}
                    persist(result, row)
                elif key not in candidate:
                    family = row.get('question_family') or row['family_id']
                    persist({'id': key, 'family_id': family, 'question_family': family,
                        'status': 'generation_missing', 'score': None, 'rounds': [], 'attempts': [],
                        'technical_error': True, 'unresolved': True, 'evaluation_key': keys[key],
                        'judgment_origin': 'generation_missing'}, row)
                else:
                    future = pool.submit(legacy._evaluate, row, docs, reference[key], candidate[key], config, secrets, keys[key])
                    futures[future] = row
            for future in as_completed(futures):
                result = future.result()
                result['judgment_origin'] = 'v6_fresh_evaluation'
                persist(result, futures[future])
        results.sort(key=lambda result: meta['ids'].index(result['id']))
        summary = summarize_scores(results, cases=rows)
        summary.update(protocol=PROTOCOL, scope='common_reference', dataset=dataset_name,
            candidate_package_sha256=candidate_meta['package_sha256'], reference_package_sha256=reference_meta['package_sha256'],
            ids=meta['ids'], status_counts=dict(Counter(r['status'] for r in results)),
            generation_valid=candidate_meta['generation_valid'], generation_invalid_reasons=candidate_meta['generation_invalid_reasons'],
            ranking_eligible=candidate_meta['generation_valid'] and reference_meta['generation_valid'],
            candidate_selection_supported=False, winner=None, not_psb_prediction=True,
            unresolved_ids=[r['id'] for r in results if r['status'] != 'ok'],
            judgment_origin='trusted_v5_cache' if cached_meta else 'v6_fresh_evaluation',
            cached_judgments=cached_meta, details_sha256=sha(output), resume_key=run_key,
            interpretation='Only completed runs may be ranked. Compare on jointly valid same cases; unresolved rows remain unknown with full-suite bounds.')
        write_json(output.with_suffix('.summary.json'), summary)
    return summary


def compare_judgments(candidate_path, incumbent_path, cases, *, repetitions=5000):
    """Checked pairwise ranking support; always descriptive, never promotion."""
    rows = load_cases(cases)
    paths = [Path(candidate_path), Path(incumbent_path)]
    metas = [json.loads(p.with_suffix('.meta.json').read_text()) for p in paths]
    summaries = [json.loads(p.with_suffix('.summary.json').read_text()) for p in paths]
    if any(m.get('protocol') != PROTOCOL or m['inputs']['cases_sha256'] != sha(cases) or m['ids'] != [r['id'] for r in rows] for m in metas):
        raise ValueError('Comparison requires exact same v6 cases')
    for key in ('semantic_protocol', 'model', 'judge_endpoint_sha256', 'prompt_sha256', 'evaluate_source_sha256', 'inputs', 'answer_orders'):
        if metas[0].get(key) != metas[1].get(key):
            raise ValueError('Comparison evaluator/source mismatch: ' + key)
    for key in ('answers_sha256', 'questions_sha256', 'run_meta_sha256', 'service_profile_sha256'):
        if metas[0]['reference'].get(key) != metas[1]['reference'].get(key):
            raise ValueError('Comparison common reference differs')
    if metas[0]['candidate']['service_profile_sha256'] != metas[1]['candidate']['service_profile_sha256']:
        raise ValueError('Comparison candidate service profiles differ')
    if any(s.get('details_sha256') != sha(p) or s.get('resume_key') != m.get('resume_key') for p, m, s in zip(paths, metas, summaries)):
        raise ValueError('Comparison judgment details changed')
    results = [csv_judge.load_rows(p) for p in paths]
    comparison = compare_systems(*results, cases=rows, scope='development', repetitions=repetitions)
    eligible = all(m['candidate']['generation_valid'] and s.get('ranking_eligible') for m, s in zip(metas, summaries))
    comparison.update(ranking_eligible=eligible, ranking_difference=comparison['observed_paired']['difference'] if eligible else None,
        promotion_supported=False, candidate_selection_supported=False,
        judgment_origins=[s['judgment_origin'] for s in summaries])
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'cases', 'kb', 'reference-run', 'candidate-run', 'services', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--reuse-judgments', type=Path)
    parser.add_argument('--reuse-cases', type=Path)
    try:
        summary = run(**vars(parser.parse_args()))
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
