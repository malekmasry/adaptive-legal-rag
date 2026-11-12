"""Rejudge every corrected label for every version, preserving all prior results.

One new two-order evaluation per changed case, irrespective of previous score.
The journal supports safe resume; unresolved results remain unresolved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from benchmark import ROOT, corpus, read_answers
from context_export import load_contexts
from judge import PROMPT, SOURCE_POLICY, complete, combine, payload_for_case, pair_payload
from labels import corrected_cases, label_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('input', 'runs', 'services', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    versions = ('vector14', 'coverage2257', 'review29', 'conditional35')
    secrets = json.loads(args.services.read_text())
    errata = json.loads((ROOT / 'label_errata.json').read_text())
    changed_ids = {row['id'] for row in errata['corrections']}
    rows = {row['id']: row for row in corrected_cases()}
    docs = corpus()
    original, metadata, answers, contexts = {}, {}, {}, {}
    baseline_path = args.runs / 'vector14' / 'submission.csv'
    baseline_contexts, baseline_context_hash = load_contexts(baseline_path.parent)
    for version in versions:
        path = args.input / (version + '.jsonl')
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        if (meta['prompt_sha256'] != hashlib.sha256(PROMPT.encode()).hexdigest()
                or meta['judge_implementation_sha256'] != digest(ROOT / 'judge.py')
                or meta['source_policy'] != SOURCE_POLICY):
            raise ValueError('Judge changed')
        if meta.get('label_errata_sha256'):
            raise ValueError('Input already has label corrections')
        items = [json.loads(line) for line in path.read_text().splitlines()]
        original[version] = {row['id']: row for row in items}
        if len(items) != len(meta['ids']) or set(original[version]) != set(meta['ids']):
            raise ValueError('Incomplete input or duplicate judgment')
        candidate_path = args.runs / version / 'submission.csv'
        if digest(candidate_path) != meta['candidate_sha256'] or digest(baseline_path) != meta['baseline_sha256']:
            raise ValueError('Answers changed')
        answers[version] = read_answers(candidate_path, meta['ids'])
        contexts[version], context_hash = load_contexts(candidate_path.parent)
        if context_hash != meta['candidate_context_sha256'] or baseline_context_hash != meta['baseline_context_sha256']:
            raise ValueError('Context changed')
        metadata[version] = {**meta, 'label_errata_sha256': label_hash(),
            'pre_errata_judgments_sha256': digest(path),
            'label_correction_ids': sorted(changed_ids & set(meta['ids'])),
            'label_rejudge_implementation_sha256': digest(Path(__file__)),
            'label_rejudge_policy': 'Exactly one two-order evaluation for every changed label in every version; no score selection.'}
        out_meta = args.output / (version + '.meta.json')
        if out_meta.exists() and json.loads(out_meta.read_text()) != metadata[version]:
            raise ValueError('Cannot mix different corrections or inputs')
        out_meta.write_text(json.dumps(metadata[version], ensure_ascii=False, indent=2) + '\n')
    baseline = answers['vector14']
    journal = args.output / 'label_correction_attempts.jsonl'
    previous = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    attempts = {(item['version'], item['id']): item for item in previous}
    jobs = [(version, key) for version in versions for key in metadata[version]['label_correction_ids']]
    if len(attempts) != len(previous) or not set(attempts) <= set(jobs):
        raise ValueError('Invalid correction journal')

    def evaluate(job):
        version, key = job
        row = rows[key]
        payload = payload_for_case(row, docs, baseline[key], answers[version][key])
        config = (secrets['LLM_BASE_URL'], secrets['LLM_API_KEY'], metadata[version]['judge_model'])
        item = {'version': version, 'id': key, 'family_id': row['family_id'], 'split': row['split'],
                'evidence_paths': sorted(payload['source_documents']), 'label_errata_sha256': label_hash()}
        rounds = []
        start = time.perf_counter()
        try:
            rounds.append(complete(pair_payload(payload, baseline[key], answers[version][key], baseline_contexts[key], contexts[version][key]), config))
            rounds.append(complete(pair_payload(payload, answers[version][key], baseline[key], contexts[version][key], baseline_contexts[key]), config))
            item.update(combine(*rounds))
        except Exception as exc:
            item.update(status='judge_error', error_type=type(exc).__name__)
        item.update(rounds=rounds, elapsed_ms=round((time.perf_counter() - start) * 1000, 2))
        return item

    with journal.open('a', buffering=1) as log, ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        for future in as_completed([pool.submit(evaluate, job) for job in jobs if job not in attempts]):
            item = future.result()
            attempts[(item['version'], item['id'])] = item
            log.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(json.dumps({key: item.get(key) for key in ('version', 'id', 'status', 'score')}), flush=True)
    for version in versions:
        with (args.output / (version + '.jsonl')).open('w') as stream:
            for key in metadata[version]['ids']:
                old = original[version][key]
                attempt = attempts.get((version, key))
                # Even an unresolved correction replaces the obsolete-label result.
                result = {**attempt, 'pre_errata_judgment': old} if attempt else old
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
