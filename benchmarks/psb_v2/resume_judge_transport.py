"""Continue incomplete judge rounds after a transport outage, once per case.

Only URLError/TimeoutError attempts with fewer than two received rounds qualify.
Completed semantic decisions, including low scores, are never retried here.
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


def incomplete_transport_attempt(row):
    if row['status'] == 'ok':
        return None
    attempt = row.get('retry_attempt', row)
    if (attempt.get('status') == 'judge_error'
            and attempt.get('error_type') in ('URLError', 'TimeoutError')
            and len(attempt.get('rounds', [])) < 2):
        return attempt
    return None


def main():
    parser = argparse.ArgumentParser()
    for name in ('input', 'runs', 'services', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    versions = ('vector14', 'coverage2257', 'review29', 'conditional35')
    rows = {row['id']: row for row in corrected_cases()}
    docs = corpus()
    secrets = json.loads(args.services.read_text())
    original, metadata, answers, contexts, pending = {}, {}, {}, {}, {}
    baseline_path = args.runs / 'vector14' / 'submission.csv'
    baseline_contexts, baseline_context_hash = load_contexts(baseline_path.parent)
    for version in versions:
        path = args.input / (version + '.jsonl')
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        if (meta['judge_implementation_sha256'] != digest(ROOT / 'judge.py')
                or meta['prompt_sha256'] != hashlib.sha256(PROMPT.encode()).hexdigest()
                or meta['source_policy'] != SOURCE_POLICY
                or meta.get('label_errata_sha256') != label_hash()):
            raise ValueError('Judge or labels changed')
        items = [json.loads(line) for line in path.read_text().splitlines()]
        original[version] = {item['id']: item for item in items}
        if len(items) != len(meta['ids']) or set(original[version]) != set(meta['ids']):
            raise ValueError('Incomplete input or duplicate judgment')
        candidate_path = args.runs / version / 'submission.csv'
        if digest(candidate_path) != meta['candidate_sha256'] or digest(baseline_path) != meta['baseline_sha256']:
            raise ValueError('Answers changed')
        answers[version] = read_answers(candidate_path, meta['ids'])
        contexts[version], context_hash = load_contexts(candidate_path.parent)
        if context_hash != meta['candidate_context_sha256'] or baseline_context_hash != meta['baseline_context_sha256']:
            raise ValueError('Context changed')
        for key, item in original[version].items():
            attempt = incomplete_transport_attempt(item)
            if attempt is not None:
                pending[(version, key)] = attempt
        metadata[version] = {**meta, 'transport_parent_sha256': digest(path),
            'transport_resume_implementation_sha256': digest(Path(__file__)),
            'transport_resume_policy': 'Once per incomplete transport-only attempt; preserve received rounds and request only missing rounds. No semantic retries.',
            'transport_resume_ids': [key for v, key in pending if v == version]}
        (args.output / (version + '.meta.json')).write_text(json.dumps(metadata[version], ensure_ascii=False, indent=2) + '\n')
    baseline = answers['vector14']

    def evaluate(job):
        version, key = job
        row = rows[key]
        rounds = list(pending[job].get('rounds', []))
        reused = len(rounds)
        payload = payload_for_case(row, docs, baseline[key], answers[version][key])
        config = (secrets['LLM_BASE_URL'], secrets['LLM_API_KEY'], metadata[version]['judge_model'])
        item = {'version': version, 'id': key, 'family_id': row['family_id'], 'split': row['split'],
                'evidence_paths': sorted(payload['source_documents']), 'reused_rounds': reused}
        start = time.perf_counter()
        try:
            if not rounds:
                rounds.append(complete(pair_payload(payload, baseline[key], answers[version][key], baseline_contexts[key], contexts[version][key]), config))
            rounds.append(complete(pair_payload(payload, answers[version][key], baseline[key], contexts[version][key], baseline_contexts[key]), config))
            item.update(combine(*rounds))
        except Exception as exc:
            item.update(status='judge_error', error_type=type(exc).__name__)
        item.update(rounds=rounds, elapsed_ms=round((time.perf_counter() - start) * 1000, 2))
        return item

    continued = {}
    with (args.output / 'transport_continuations.jsonl').open('w', buffering=1) as log, ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        for future in as_completed([pool.submit(evaluate, job) for job in pending]):
            item = future.result()
            continued[(item['version'], item['id'])] = item
            log.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(json.dumps({key: item.get(key) for key in ('version', 'id', 'status', 'score', 'reused_rounds')}), flush=True)
    for version in versions:
        with (args.output / (version + '.jsonl')).open('w') as stream:
            for key in metadata[version]['ids']:
                old = original[version][key]
                new = continued.get((version, key))
                result = {**new, 'before_transport_resume': old} if new else old
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
