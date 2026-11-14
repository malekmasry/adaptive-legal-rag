"""Compare independently scored answers; diagnose saturation before optimization.

Near variants are dependent observations. Bootstrap resamples entire source
clusters, preserving every question and contrast pair inside the sampled unit.
These intervals describe this small local dataset, not unseen PSB performance.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics

from v8_absolute_judge import object_sha, sha, write_json

SEED = 2026111408
REPEATS = 4000
CEILING_FRACTION = .85


def _quantile(values, p):
    values = sorted(values)
    return values[int((len(values) - 1) * p)]


def paired(left, right, cases, repeats=REPEATS):
    a = {r['id']: r for r in left}; b = {r['id']: r for r in right}
    groups = defaultdict(list)
    for case in cases:
        key = case['id']
        cluster = case.get('source_cluster', case.get('cluster_id'))
        if not cluster:
            raise ValueError('A source cluster is required; question independence must not be assumed')
        x = a.get(key, {'score': None, 'bounds': [0, 100]})
        y = b.get(key, {'score': None, 'bounds': [0, 100]})
        point = None if x.get('score') is None or y.get('score') is None else y['score'] - x['score']
        groups[cluster].append({'id': key, 'source_cluster': cluster, 'delta': point,
                               'bounds': [y['bounds'][0] - x['bounds'][1], y['bounds'][1] - x['bounds'][0]]})
    rows = [r for g in groups.values() for r in g]
    if not rows:
        raise ValueError('No comparison cases')
    complete = all(r['delta'] is not None for r in rows)
    names = sorted(groups); rng = random.Random(SEED)
    point_samples = []; lows = []; highs = []
    for _ in range(repeats):
        sample = [r for _ in names for r in groups[rng.choice(names)]]
        lows.append(statistics.mean(r['bounds'][0] for r in sample))
        highs.append(statistics.mean(r['bounds'][1] for r in sample))
        if complete:
            point_samples.append(statistics.mean(r['delta'] for r in sample))
    interval = [_quantile(point_samples, .025), _quantile(point_samples, .975)] if complete and len(names) > 1 else None
    conservative = [_quantile(lows, .025), _quantile(highs, .975)] if len(names) > 1 else None
    return {'direction': 'right_minus_left', 'count': len(rows), 'source_clusters': len(names),
            'all_questions_delta': statistics.mean(r['delta'] for r in rows) if complete else None,
            'all_question_bounds': [statistics.mean(r['bounds'][i] for r in rows) for i in (0, 1)],
            'source_cluster_bootstrap_95': interval,
            'conservative_source_cluster_bootstrap_95': conservative,
            'seed': SEED, 'repeats': repeats, 'per_case': rows,
            'small_cluster_warning': len(names) < 10,
            'limitation': 'Exploratory paired interval, not a PSB prediction or a multiple-comparison corrected winner test.'}


def dataset_health(systems, pairs, cases):
    """No winner and no impossible fixed +5 gate. Unknowns never disappear."""
    n = len(cases); diagnostics = {}
    for name, rows in systems.items():
        observed = [r for r in rows if r.get('score') is not None]
        complete = len(observed) == n
        perfect = sum(r['score'] == 100 for r in observed)
        diagnostics[name] = {'scored': len(observed), 'expected': n,
            'mean_score': statistics.mean(r['score'] for r in observed) if complete else None,
            'remaining_score_headroom': 100 - statistics.mean(r['score'] for r in observed) if complete else None,
            'perfect_cases': perfect,
            'perfect_fraction_all_cases': perfect / n if complete else None,
            'perfect_fraction_bounds': [perfect / n, (perfect + n - len(observed)) / n],
            'ceiling_warning': complete and perfect / n >= CEILING_FRACTION}
    all_scored = all(d['scored'] == n for d in diagnostics.values())
    separated = []
    for p in pairs:
        interval = p['conservative_source_cluster_bootstrap_95']
        if interval is not None and (interval[0] > 0 or interval[1] < 0):
            separated.append([p['left'], p['right']])
    return {'all_scored': all_scored, 'systems': diagnostics,
            'ceiling_warning_fraction': CEILING_FRACTION,
            'exploratory_separated_pairs': separated,
            'discrimination_observed': all_scored and bool(separated),
            'automatic_optimization_allowed': False,
            'selection_decision': None,
            'interpretation': 'Separation is a diagnostic. Review label/judge controls and errors first; no automatic architecture winner or target organizer score.'}


def contrast_diagnostics(systems, cases):
    groups = defaultdict(list)
    for case in cases:
        key = case.get('contrast_pair_id', case.get('contrast_group'))
        if key:
            groups[key].append(case['id'])
    report = {}
    for name, rows in systems.items():
        by_id = {r['id']: r for r in rows}
        values = []
        for group, ids in sorted(groups.items()):
            scores = [by_id.get(key, {}).get('score') for key in ids]
            values.append({'group': group, 'ids': ids, 'scores': scores,
                           'all_members_scored': all(x is not None for x in scores),
                           'members_at_100': sum(x == 100 for x in scores),
                           'all_members_at_100': all(x == 100 for x in scores) if all(x is not None for x in scores) else None})
        report[name] = values
    return {'systems': report, 'note': 'Answers are judged separately against their own sources and facts; no demand for different wording and no answer-vs-answer judge.'}


def load_result(path, candidate, cases_path, corpus, *, allow_test=False):
    import v8_absolute_judge as judge
    path = Path(path); mp = path.with_suffix('.meta.json')
    meta = json.loads(mp.read_text()); saved = json.loads(path.with_suffix('.summary.json').read_text())
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if meta.get('protocol') != judge.PROTOCOL or saved.get('details_sha256') != sha(path) or saved.get('meta_sha256') != sha(mp):
        raise ValueError('Judgment artifact identity differs')
    if meta.get('rubric') != judge.RUBRIC or meta.get('prompt_sha256') != hashlib.sha256(judge.PROMPT.encode()).hexdigest() or meta.get('packing') != judge.PACKING:
        raise ValueError('Current judge rubric or prompt differs from judged version')
    if meta.get('implementation') != {'v8_absolute_judge.py': sha(judge.__file__)}:
        raise ValueError('Current judge implementation differs from judged version')
    if meta.get('execution_mode') != 'live_api' and not allow_test:
        raise ValueError('Offline test caller output cannot be used as live benchmark evidence')
    if meta['inputs'] != {'candidate': sha(candidate), 'cases': sha(cases_path), 'corpus': sha(corpus)}:
        raise ValueError('Comparison inputs differ from judged inputs')
    if object_sha({k: v for k, v in meta.items() if k != 'resume_key'}) != meta['resume_key']:
        raise ValueError('Invalid judgment metadata digest')
    docs = judge.read_docs(corpus); cases = judge.load_cases(cases_path, docs)
    answers = judge.read_answers(candidate, [c['id'] for c in cases]); by_id = {c['id']: c for c in cases}
    if meta['ids'] != list(by_id):
        raise ValueError('Judgment IDs differ from frozen dataset')
    for row in rows:
        if row['id'] not in by_id:
            raise ValueError('Unrelated judgment row')
        case = by_id[row['id']]; answer = answers.get(row['id'], '')
        key = object_sha({'run': meta['resume_key'], 'case': case, 'answer': answer})
        judge._validate_saved(row, case, answer, docs, key,
                              expected_model=meta['model'], require_live=not allow_test)
    summary = judge.summarize(rows, meta['ids'], cases)
    if any(saved.get(k) != v for k, v in summary.items()):
        raise ValueError('Saved summary arithmetic differs')
    return meta, rows, summary, cases


def compare(paths, answers, cases, corpus, *, allow_test=False):
    if set(paths) != set(answers) or len(paths) < 2:
        raise ValueError('Each of at least two systems needs judgments and its exact answer CSV')
    loaded = {name: load_result(path, answers[name], cases, corpus, allow_test=allow_test) for name, path in paths.items()}
    def identity(meta):
        keys = ('protocol', 'rubric', 'packing', 'execution_mode', 'model', 'scope', 'ids', 'passes', 'prompt_sha256',
                'implementation', 'judge_endpoint_sha256', 'credential_fingerprint')
        return {k: meta[k] for k in keys} | {'cases': meta['inputs']['cases'], 'corpus': meta['inputs']['corpus']}
    first = next(iter(loaded.values()))
    if any(identity(item[0]) != identity(first[0]) for item in loaded.values()):
        raise ValueError('Cases, source, rubric, model or judge implementation differs')
    pairs = [{'left': a, 'right': b, **paired(loaded[a][1], loaded[b][1], first[3])}
             for a, b in itertools.combinations(loaded, 2)]
    return {'protocol': first[0]['protocol'], 'scope': first[0]['scope'], 'offline_fixture_mode': allow_test,
            'systems': {n: v[2] for n, v in loaded.items()}, 'pairs': pairs,
            'health': dataset_health({n: v[1] for n, v in loaded.items()}, pairs, first[3]),
            'contrast_diagnostics': contrast_diagnostics({n: v[1] for n, v in loaded.items()}, first[3]),
            'complete': all(v[2]['complete'] for v in loaded.values()),
            'all_scored': all(v[2]['all_scored'] for v in loaded.values()),
            'selection_decision': None, 'organizer_equivalence': False,
            'source_sha256': {n: sha(p) for n, p in paths.items()}}


def specs(values):
    result = {}
    for value in values:
        name, path = value.split('=', 1)
        if not name or name in result:
            raise ValueError('Empty/duplicate system')
        result[name] = Path(path)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system', action='append', required=True, help='name=judgments.jsonl')
    p.add_argument('--answer', action='append', required=True, help='name=submission.csv')
    for key in ('cases', 'kb', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args(); result = compare(specs(a.system), specs(a.answer), a.cases, a.kb)
    a.out.parent.mkdir(parents=True, exist_ok=True); write_json(a.out, result)
    print(json.dumps({'complete': result['complete'], 'all_scored': result['all_scored'], 'health': result['health']}, ensure_ascii=False))
