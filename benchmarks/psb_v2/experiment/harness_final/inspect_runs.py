"""Summarize actual calls and expose every answer/judgment for manual inspection."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from benchmark import cases


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--judgments', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='development')
    args = parser.parse_args()
    versions = ('vector14', 'coverage2257', 'review29', 'conditional35')
    summary, answers, judgments = {}, {}, {}
    case_index={row['id']:row for row in cases(args.split)}
    for version in versions:
        directory = args.runs / version
        meta = json.loads((directory / 'run.meta.json').read_text())
        rows = [row for row in json_lines(directory / 'submission.metrics.jsonl') if 'id' in row]
        calls = [call for row in rows for call in row.get('api_calls', [])]
        evaluation = json_lines(args.judgments / (version + '.jsonl'))
        timings = [row['total_ms'] / 1000 for row in rows if 'total_ms' in row]
        judge_rounds = [round_ for row in evaluation for round_ in row.get('rounds', [])]
        usage = [round_.get('_api', {}).get('usage', {}) for round_ in judge_rounds]
        summary[version] = {
            'commit': meta['commit'], 'elapsed_seconds': meta.get('elapsed_seconds'),
            'questions': len(rows), 'statuses': dict(Counter(row.get('status') for row in rows)),
            'median_question_seconds': statistics.median(timings) if timings else None,
            'calls_by_stage': dict(Counter(call.get('stage', 'unknown') for call in calls)),
            'calls_by_endpoint': dict(Counter(call.get('endpoint', 'unknown') for call in calls)),
            'http_statuses': dict(Counter(str(call.get('http_status', 'not_reported')) for call in calls)),
            'reported_prompt_tokens': sum(row.get('prompt_tokens', 0) for row in rows),
            'reported_completion_tokens': sum(row.get('completion_tokens', 0) for row in rows),
            'token_usage_complete': all(row.get('usage_complete', False) for row in rows),
            'events': dict(Counter(event.get('stage', 'unknown') for row in rows for event in row.get('events', []))),
            'event_outcomes': dict(Counter(event.get('stage', 'unknown') + ':' + str(event.get('status', ''))
                                           for row in rows for event in row.get('events', []))),
            'questions_with_dense_search': sum(any(event.get('dense_used') is True for event in row.get('events', [])) for row in rows),
            'transport_failure_ids': [row['id'] for row in rows if any(
                call.get('stage') not in ('answer_review','evidence_coverage') and (call.get('error_type') or (call.get('http_status') or 200)>=400)
                for call in row.get('api_calls',[]))],
            'api_failures_by_stage': dict(Counter(call.get('stage','unknown') for call in calls
                if call.get('error_type') or (call.get('http_status') or 200)>=400)),
            'review_by_question_origin': dict(Counter(
                case_index[row['id']].get('origin','unknown')+':'+str(event.get('status'))
                for row in rows for event in row.get('events',[]) if event.get('stage')=='answer_review')),
            'service_observations': meta.get('service_observations'),
            'judge_statuses': dict(Counter(row['status'] for row in evaluation)),
            'confirmed_hallucination_ids': [row['id'] for row in evaluation if row.get('confirmed_hallucination')],
            'unresolved_ids': [row['id'] for row in evaluation if row['status'] != 'ok'],
            'large_order_disagreement_ids': [row['id'] for row in evaluation if row.get('order_disagreement', 0) >= .5],
            'mean_order_disagreement': statistics.mean([row.get('order_disagreement', 0) for row in evaluation
                                                       if row['status'] == 'ok']) if any(row['status']=='ok' for row in evaluation) else None,
            'judge_reported_prompt_tokens': sum(item.get('prompt_tokens', 0) for item in usage),
            'judge_reported_completion_tokens': sum(item.get('completion_tokens', 0) for item in usage),
            'judge_usage_caveat': 'Only retained successful rounds; failed attempts may consume additional tokens.',
        }
        with (directory / 'submission.csv').open() as stream:
            answers[version] = {row['id']: row['answer'] for row in csv.DictReader(stream)}
        judgments[version] = {row['id']: row for row in evaluation}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'runtime_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'question_comparison.jsonl').open('w') as stream:
        for row in cases(args.split):
            item = {key: row[key] for key in ('id', 'family_id', 'category', 'question', 'reference_answer', 'required_facts', 'evidence')}
            item['versions'] = {version: {'answer': answers[version].get(row['id']),
                                         'judgment': judgments[version].get(row['id'])} for version in versions}
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(json.dumps({version: {'questions': row['questions'], 'elapsed_seconds': row['elapsed_seconds'],
                               'judge_statuses': row['judge_statuses']} for version, row in summary.items()}))


if __name__ == '__main__':
    main()
