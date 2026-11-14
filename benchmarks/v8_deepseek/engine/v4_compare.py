"""Compare systems independently judged against one fixed reference.

The caller must enforce identical reference answers, judge configuration, and
case definitions. This module compares their PSB-style aggregate scores, not a
direct candidate-versus-incumbent judgment. ``score``/``family_id`` are the
existing csv_judge schema; ``mean_score`` and ``question_family``/``family`` are
accepted aliases. Unknown judgments remain unknown in the observed report.

Pass the complete case list explicitly when a result file may be incomplete.
Without it, both result lists must contain exactly the same IDs. A supported
full-control comparison is only one promotion prerequisite: the outer
controller must also compare every required incumbent and pass security gates.
"""

if __name__ == '__main__':
    raise SystemExit('Historical dependency only. Use v8_run.py in the prepared DeepSeek benchmark.')


import math
import random
from collections import defaultdict
from collections.abc import Mapping


DEFAULT_REPETITIONS = 5000
DEFAULT_SEED = 2026111402
FULL_CONTROL_SCOPES = frozenset({'full_control', 'heldout'})
FAMILY_FIELDS = ('family_id', 'question_family', 'family')


def _family(row, required=True):
    values = [row[key] for key in FAMILY_FIELDS if key in row]
    if not values:
        if required:
            raise ValueError('Missing family for ID: ' + row['id'])
        return None
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError('Invalid family for ID: ' + row['id'])
    if len(set(values)) != 1:
        raise ValueError('Conflicting family aliases for ID: ' + row['id'])
    return values[0]


def _index(rows, label):
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(label + ' rows must be mappings')
        key = row.get('id')
        if not isinstance(key, str) or not key.strip():
            raise ValueError(label + ' IDs must be nonempty strings')
        if key in result:
            raise ValueError('Duplicate ' + label + ' ID: ' + key)
        result[key] = row
    return result


def _score(row):
    if row.get('status') != 'ok':
        return None
    values = [row[key] for key in ('score', 'mean_score') if key in row]
    if not values or any(type(value) not in (int, float) or
                         not math.isfinite(value) or not 0 <= value <= 1
                         for value in values):
        raise ValueError('An ok result needs a finite score in [0, 1]: ' + row['id'])
    if len(set(values)) != 1:
        raise ValueError('Conflicting score aliases for ID: ' + row['id'])
    return float(values[0])


def _align(result_sets, cases):
    indexed = [_index(rows, 'result') for rows in result_sets]
    if cases is None:
        expected_ids = set(indexed[0])
        if any(set(rows) != expected_ids for rows in indexed[1:]):
            raise ValueError('Result ID sets differ; supply the complete case list for missing results')
        case_index = indexed[0]
    else:
        case_index = _index(cases, 'case')
        expected_ids = set(case_index)
        if any(set(rows) - expected_ids for rows in indexed):
            raise ValueError('Unexpected result IDs outside the complete case list')
    if not expected_ids:
        raise ValueError('The complete case list must not be empty')
    normalized = [[] for _ in indexed]
    for key in sorted(expected_ids):
        family = _family(case_index[key])
        for output, rows in zip(normalized, indexed):
            row = rows.get(key)
            if row is None:
                output.append({'id': key, 'family_id': family, 'score': None})
                continue
            actual_family = _family(row, required=cases is None)
            if actual_family is not None and actual_family != family:
                raise ValueError('Family mismatch for ID: ' + key)
            output.append({'id': key, 'family_id': family, 'score': _score(row)})
    return normalized


def _stats(scores):
    values = [score for score in scores if score is not None]
    return (len(values), math.fsum(values), sum(score > .5 for score in values),
            sum(score < .5 for score in values))


def _from_stats(stats):
    count, total, wins, losses = stats
    if not count:
        return {'score': None, 'match_rate': None, 'win_rate': None,
                'wins': 0, 'losses': 0, 'ties': 0, 'evaluated': 0}
    mr = total / count
    wr = wins / (wins + losses) if wins + losses else .5
    return {'score': .7 * mr + .3 * wr, 'match_rate': mr, 'win_rate': wr,
            'wins': wins, 'losses': losses, 'ties': count - wins - losses,
            'evaluated': count}


def _aggregate(scores):
    result = _from_stats(_stats(scores))
    result.update(attempted=len(scores), coverage=result['evaluated'] / len(scores)
                  if scores else 0)
    return result


def _summary(rows):
    scores = [row['score'] for row in rows]
    result = _aggregate(scores)
    result['unknown_count'] = len(scores) - result['evaluated']
    result['unknown_score_bounds'] = {
        'low': _aggregate([value if value is not None else 0 for value in scores])['score'],
        'high': _aggregate([value if value is not None else 1 for value in scores])['score'],
    }
    result['score_basis'] = 'Observed ok judgments only; unknowns excluded from published MR and WR'
    return result


def summarize_scores(rows, cases=None):
    """Return observed PSB-style score, coverage, and full-suite unknown bounds.

    Missing rows are unknown only when ``cases`` supplies the complete suite.
    Malformed ok scores, duplicate IDs, and conflicting metadata raise ValueError.
    """
    return _summary(_align([rows], cases)[0])


def _quantile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _bootstrap(pairs, repetitions, seed):
    """Draw paired whole families, preserving question weights inside families."""
    groups = defaultdict(list)
    for candidate, incumbent in pairs:
        groups[candidate['family_id']].append((candidate['score'], incumbent['score']))
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    # Sufficient statistics preserve the score's nonlinear tie-excluding WR;
    # averaging per-case differences would not compute this aggregate delta.
    family_stats = []
    for family in keys:
        scores = groups[family]
        family_stats.append([
            _stats([c if c is not None else 0 for c, _ in scores]),
            _stats([i if i is not None else 1 for _, i in scores]),
            _stats([c if c is not None else 1 for c, _ in scores]),
            _stats([i if i is not None else 0 for _, i in scores]),
        ])
    rng = random.Random(seed)
    pessimistic, optimistic = [], []
    for _ in range(repetitions):
        sample = rng.choices(family_stats, k=len(keys))
        values = [_from_stats(tuple(math.fsum(family[index][column] for family in sample)
                                   for column in range(4)))['score'] for index in range(4)]
        pessimistic.append(values[0] - values[1])
        optimistic.append(values[2] - values[3])
    pessimistic.sort()
    optimistic.sort()
    return {
        'low': _quantile(pessimistic, .025),
        'high': _quantile(optimistic, .975),
        'method': 'paired family-cluster percentile bootstrap with unknown-score envelope',
        'independent_families': len(keys), 'repetitions': repetitions, 'seed': seed,
        'lower_assignment': 'candidate unknown=0; incumbent unknown=1',
        'upper_assignment': 'candidate unknown=1; incumbent unknown=0',
    }


def compare_systems(candidate_rows, incumbent_rows, *, cases=None, scope='pilot',
                    repetitions=DEFAULT_REPETITIONS, seed=DEFAULT_SEED):
    """Return candidate-minus-incumbent common-reference aggregate differences.

    ``promotion_supported`` is always false for pilots. On full_control or
    heldout it requires >=90% coverage for *each* system and a strictly positive
    lower conservative 95% bootstrap limit. This is not an overall winner or
    organizer-score prediction; all other incumbent/security gates belong to
    the caller. At least two independent families are required for an interval.
    """
    if type(repetitions) is not int or repetitions < 2000:
        raise ValueError('At least 2000 bootstrap repetitions are required')
    if type(seed) is not int:
        raise ValueError('Bootstrap seed must be an integer')
    if scope not in {'pilot', 'development', 'full_development'} | FULL_CONTROL_SCOPES:
        raise ValueError('Unsupported comparison scope: ' + str(scope))
    candidate, incumbent = _align([candidate_rows, incumbent_rows], cases)
    candidate_summary, incumbent_summary = _summary(candidate), _summary(incumbent)
    pairs = list(zip(candidate, incumbent))
    valid_pairs = [(c, i) for c, i in pairs if c['score'] is not None and i['score'] is not None]
    paired_candidate = _aggregate([c['score'] for c, _ in valid_pairs])
    paired_incumbent = _aggregate([i['score'] for _, i in valid_pairs])
    paired_delta = (paired_candidate['score'] - paired_incumbent['score']) if valid_pairs else None
    observed_interval = _bootstrap(valid_pairs, repetitions, seed)
    if observed_interval:
        observed_interval['method'] = 'paired family-cluster percentile bootstrap on jointly valid cases'
        observed_interval.pop('lower_assignment')
        observed_interval.pop('upper_assignment')
    conservative_interval = _bootstrap(pairs, repetitions, seed)
    bounds = {
        'low': candidate_summary['unknown_score_bounds']['low'] - incumbent_summary['unknown_score_bounds']['high'],
        'high': candidate_summary['unknown_score_bounds']['high'] - incumbent_summary['unknown_score_bounds']['low'],
    }
    observed_delta = None
    if candidate_summary['score'] is not None and incumbent_summary['score'] is not None:
        observed_delta = candidate_summary['score'] - incumbent_summary['score']
    coverage_ok = min(candidate_summary['coverage'], incumbent_summary['coverage']) >= .9
    supported = bool(scope in FULL_CONTROL_SCOPES and coverage_ok and
                     conservative_interval and conservative_interval['low'] > 0)
    return {
        'scope': scope, 'comparison': 'candidate minus incumbent; each judged versus the same fixed reference',
        'candidate': candidate_summary, 'incumbent': incumbent_summary,
        'observed_difference': observed_delta,
        'observed_difference_basis': 'Each system uses its own observed ok cases; see observed_paired for matched coverage',
        'observed_paired': {
            'difference': paired_delta, 'candidate': paired_candidate, 'incumbent': paired_incumbent,
            'evaluated': len(valid_pairs), 'attempted': len(pairs),
            'coverage': len(valid_pairs) / len(pairs),
            'ids': [c['id'] for c, _ in valid_pairs],
            'family_bootstrap_95': observed_interval,
        },
        'unknown_difference_bounds': bounds,
        'conservative_family_bootstrap_95': conservative_interval,
        'promotion_supported': supported,
        'promotion_gate': {'full_control': scope in FULL_CONTROL_SCOPES,
                           'both_coverage_at_least_90_percent': coverage_ok,
                           'conservative_lower_above_zero': bool(conservative_interval and conservative_interval['low'] > 0)},
        'not_psb_prediction': True,
        'warning': 'Local suite resampling variability only; independent replication and other incumbent/security gates are separate.',
    }
