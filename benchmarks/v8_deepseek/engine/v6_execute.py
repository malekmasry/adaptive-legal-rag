"""Run one frozen v6 ZIP. No gold, tests, or private config enter its directory.

Freeze plan.packages[name]={package,sha256,files,family}, datasets[name]=
{cases_sha256,count}, corpus_sha256, service_profile_sha256, workers=3 and
implementations before invoking this command. Existing output directories are
never reused. Additional modules need an explicit extra_modules entry as well
as their exact files hash. Appending other packages does not change run identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from zipfile import ZipFile

from embedding_cache import EmbeddingCache, MODEL, REVISION
from live_embedding_server import LazyBGE, make_server
from run_matrix import ENV_NAMES, RUNTIME_FILES
from v5_common_judge import load_cases, object_sha, sha, write_json

ROOT = Path(__file__).resolve().parent
IMPLEMENTATIONS = ('v6_execute.py', 'v6_common_judge.py', 'v5_common_judge.py',
                   'csv_judge.py', 'judge.py', 'benchmark.py', 'selection.py',
                   'run_matrix.py', 'v4_compare.py', 'embedding_cache.py',
                   'live_embedding_server.py', 'bge_weights.json')
V8_FILES = frozenset(('solution.py settings.py schemas.py security.py corpus.py storage.py '
    'model_api.py retriever35.py retriever35_core.py retrieval.py reasoning.py pipeline.py '
    'source_audit.py token_budget.py prompt_config.py diagnostics.py prompts.yaml aliases.yaml '
    'relations.yaml requirements.txt cl100k_base.tiktoken').split())
BRIDGES = {'query_bridge.py', 'evidence_bridge.py', 'review_bridge.py'}
COMPLETE_STATUSES = {'answered', 'partial', 'clarification', 'not_in_base', 'insufficient'}
BAD_STATUSES = {'technical_error', 'initialization_error', 'deadline_inflight', 'deadline_unprocessed'}
OBSERVATION_FILES = ('submission.metrics.jsonl', 'cache/diagnostics.jsonl',
                     'submission.report.json', 'embedding.requests.jsonl')


def implementation_hashes(root=ROOT):
    return {name: sha(root / name) for name in IMPLEMENTATIONS}


def public_settings(config):
    public = {k: v for k, v in config.items() if not k.endswith('_KEY') and not k.endswith('_BASE_URL')}
    public.update(EMB_MODEL=MODEL, embedding_provider='local pinned CLS BGE-M3', embedding_revision=REVISION)
    public['endpoint_fingerprints'] = {k: hashlib.sha256(v.encode()).hexdigest()
        for k, v in config.items() if k.endswith('_BASE_URL') and not k.startswith('EMB_')}
    return public


def profile_sha(settings):
    # Deliberately byte-compatible with frozen v5 generation profiles.
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_plan(path, *, check_implementations=True, root=ROOT):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('workers') != 3 or not isinstance(plan.get('packages'), dict):
        raise ValueError('Plan must freeze packages and workers=3')
    if check_implementations and plan.get('implementations') != implementation_hashes(root):
        raise ValueError('Frozen harness implementation hashes differ')
    return plan


def dataset_entry(plan, cases, rows, kb):
    if sha(kb) != plan.get('corpus_sha256'):
        raise ValueError('Corpus differs from frozen plan')
    matches = [(n, d) for n, d in plan.get('datasets', {}).items() if d.get('cases_sha256') == sha(cases)]
    if len(matches) != 1 or matches[0][1].get('count') != len(rows):
        raise ValueError('Cases/count absent from frozen plan')
    return matches[0]


def _flat(name):
    return isinstance(name, str) and bool(name) and name not in {'.', '..'} and '/' not in name and '\\' not in name


def verify_archive(package, entry):
    files = entry.get('files', {})
    family = entry.get('family')
    if family not in {'selected', 'v8'} or not isinstance(files, dict) or 'solution.py' not in files:
        raise ValueError('Invalid package family or files')
    extras = entry.get('extra_modules', [])
    if not isinstance(extras, list) or any(not _flat(n) or not n.endswith('.py') for n in extras):
        raise ValueError('Invalid explicit extra module allowlist')
    allowed = set(RUNTIME_FILES) | BRIDGES | set(extras) if family == 'selected' else set(V8_FILES) | BRIDGES | set(extras)
    for name in files:
        if not _flat(name) or re.search(r'(^|[_\-.])(test|tests|gold|private|services|cases|judge|benchmark)([_\-.]|$)', name, re.I):
            raise ValueError('Non-runtime or non-flat package member')
        is_cache = family == 'v8' and re.fullmatch(r'(vectors35-[a-f0-9]{64}\.npz|source-audit-[a-f0-9]{64}\.json)', name)
        if name not in allowed and not is_cache:
            raise ValueError('Package member absent from runtime allowlist: ' + name)
    if sha(package) != entry.get('sha256'):
        raise ValueError('Frozen package SHA differs')
    with ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError('Frozen package members differ')
        for info in archive.infolist():
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive links/directories forbidden')
            if hashlib.sha256(archive.read(info.filename)).hexdigest() != files[info.filename]:
                raise ValueError('Frozen package contents differ')
    return files


def verify_runtime(directory, files):
    directory = Path(directory)
    if {p.name for p in directory.glob('*.py')} != {n for n in files if n.endswith('.py')}:
        raise ValueError('Runtime Python modules differ from frozen archive')
    for name, expected in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError('Runtime file changed: ' + name)


def read_emitted_answers(path, ids):
    """Read exact CSV bytes without inventing answers for missing/empty rows."""
    path = Path(path)
    if not path.exists():
        return {}, {'missing_ids': list(ids), 'empty_ids': [], 'invalid_ids': [], 'csv_present': False}
    result, seen, empty, invalid = {}, set(), [], []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'answer']:
            raise ValueError('CSV columns must be exactly id,answer')
        for row in reader:
            if set(row) != {'id', 'answer'} or row['id'] in seen or row['id'] not in ids or row['answer'] is None:
                raise ValueError('Malformed, duplicate or unrelated answer CSV row')
            key, answer = row['id'], row['answer']
            seen.add(key)
            if not answer.strip():
                empty.append(key)
            elif len(answer) > 10000:
                invalid.append(key)
            else:
                result[key] = answer
    return result, {'missing_ids': [key for key in ids if key not in seen], 'empty_ids': empty,
                    'invalid_ids': invalid, 'csv_present': True}


def _events(path):
    if not path.exists():
        return [], []
    events, errors = [], []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
            events.append(row)
        except (ValueError, TypeError):
            errors.append({'file': path.name, 'line': number})
    return events, errors


def observe(directory, family, ids, files):
    """Recompute observations from this run's logs; calls are not answers."""
    directory = Path(directory)
    errors, duplicates = [], []
    if family == 'selected':
        events, errors = _events(directory / 'submission.metrics.jsonl')
        answered = [row for row in events if 'id' in row]
        statuses, calls = {}, []
        for row in answered:
            key = row['id']
            if key in statuses:
                duplicates.append(key)
            statuses[key] = row.get('status')
            calls.extend({**c, 'request_id': key} for c in row.get('api_calls', []) if isinstance(c, dict))
        init = next((e for e in events if e.get('stage') == 'initialization'), {})
        dense = init.get('dense_index_ready')
        if dense is None:
            dense = any(e.get('dense_used') is True for row in answered for e in row.get('events', []))
        rerank = lambda c: c.get('endpoint') == '/rerank' and c.get('http_status') == 200
        generation = lambda c: c.get('endpoint') == '/chat/completions' and c.get('http_status') == 200
        failed = lambda c: c.get('http_status') != 200
        degraded = 0
    else:
        events, errors = _events(directory / 'cache/diagnostics.jsonl')
        report_path = directory / 'submission.report.json'
        try:
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            statuses = report.get('statuses', {})
            if not isinstance(statuses, dict):
                raise ValueError()
        except (ValueError, TypeError):
            statuses = {}
            errors.append({'file': report_path.name, 'line': 0})
        calls = [e for e in events if e.get('event') == 'api_call']
        dense = any(n.startswith('vectors35-') for n in files) and not any(
            e.get('event') in {'initialization_failed', 'initialization_degraded'} and
            (e.get('event') == 'initialization_failed' or e.get('component') == 'embeddings') for e in events)
        rerank = lambda c: c.get('kind') == 'rerank' and c.get('status') == 'ok'
        generation = lambda c: c.get('kind') == 'chat' and c.get('status') == 'ok'
        failed = lambda c: c.get('status') != 'ok'
        degraded = sum(e.get('event') == 'answer_completed' and e.get('degraded') is True for e in events)
    completed = [key for key in ids if statuses.get(key) in COMPLETE_STATUSES]
    per_id_calls = {key: sum(generation(c) and c.get('request_id') == key for c in calls) for key in ids}
    obs = {'dense_index_ready': bool(dense), 'rerank_success': sum(rerank(c) for c in calls),
        'generation_success': sum(generation(c) for c in calls), 'completed_answers': len(completed),
        'completed_ids': completed, 'technical_errors': sum(statuses.get(k) in BAD_STATUSES for k in ids),
        'statuses': dict(sorted(Counter(str(v) for v in statuses.values()).items())),
        'answer_statuses': statuses, 'missing_status_ids': [k for k in ids if k not in statuses],
        'unknown_status_ids': [k for k in ids if k in statuses and statuses[k] not in COMPLETE_STATUSES | BAD_STATUSES],
        'unrelated_status_ids': sorted(set(statuses) - set(ids)), 'duplicate_status_ids': duplicates,
        'generation_success_by_id': per_id_calls, 'degraded_answers': degraded,
        'api_failed_calls': sum(failed(c) for c in calls), 'log_parse_errors': errors}
    usage = {key: sum(c.get('usage', {}).get(key, c.get(key, 0)) or 0 for c in calls)
             for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
    return obs, usage, len(calls)


def assessment(meta, obs, answers, issues, ids):
    reasons = []
    if meta.get('exit_code') != 0:
        reasons.append('nonzero_or_missing_exit')
    if meta.get('outer_timeout'):
        reasons.append('outer_timeout')
    if set(answers) != set(ids) or any(issues.get(k) for k in ('empty_ids', 'invalid_ids', 'missing_ids')):
        reasons.append('incomplete_answer_csv')
    if obs['completed_answers'] != len(ids) or obs['technical_errors']:
        reasons.append('incomplete_generation')
    if any(obs[k] for k in ('missing_status_ids', 'unknown_status_ids', 'unrelated_status_ids', 'duplicate_status_ids', 'log_parse_errors')):
        reasons.append('incomplete_or_invalid_telemetry')
    is_reference = meta.get('evaluation_role') == 'fixed_bm25_reference'
    if is_reference:
        retrieval = not obs['dense_index_ready'] and not obs['rerank_success']
    else:
        retrieval = obs['dense_index_ready'] and obs['rerank_success'] > 0
    if not retrieval:
        reasons.append('retrieval_service_not_observed')
    required_ids = [key for key in ids if obs['answer_statuses'].get(key) in {'answered', 'partial'}]
    if any(obs['generation_success_by_id'].get(key, 0) < 1 for key in required_ids):
        reasons.append('insufficient_successful_generator_calls')
    return {'valid_run': not reasons, 'all_ids_valid': set(answers) == set(ids),
            'invalid_reasons': reasons, 'answer_csv_issues': issues}


def observation_hashes(directory):
    return {name: sha(Path(directory) / name) for name in OBSERVATION_FILES if (Path(directory) / name).exists()}


def run(a):
    plan = load_plan(a.plan)
    rows = load_cases(a.cases)
    dataset_name, dataset = dataset_entry(plan, a.cases, rows, a.kb)
    entry = plan['packages'][a.name]
    verify_archive(a.package, entry)
    config = json.loads(a.services.read_text())
    if set(config) - (ENV_NAMES | {'AS_OF', 'EMBEDDER_BASE_URL', 'EMBEDDER_API_KEY', 'GENERATOR_MODEL', 'GENERATOR_BASE_URL', 'GENERATOR_API_KEY', 'JUDGE_MODEL', 'JUDGE_BASE_URL', 'JUDGE_API_KEY'}) or any(not isinstance(v, str) for v in config.values()):
        raise ValueError('Unknown or non-string private service setting')
    from model_roles import validate_roles
    validate_roles(config)
    if config.get('MAX_WORKERS') != '3':
        raise ValueError('Private service profile must use MAX_WORKERS=3')
    public = public_settings(config)
    if profile_sha(public) != plan.get('service_profile_sha256'):
        raise ValueError('Service profile differs from frozen plan')
    random.Random(a.seed).shuffle(rows)
    ids = [row['id'] for row in rows]
    a.output.mkdir(parents=True, exist_ok=False)
    data = a.output / 'data'
    data.mkdir()
    with ZipFile(a.package) as archive:
        archive.extractall(a.output)
    with (data / 'questions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'question'])
        writer.writeheader()
        writer.writerows({'id': row['id'], 'question': row['question']} for row in rows)
    (data / 'kb.zip').write_bytes(a.kb.read_bytes())
    meta = {'protocol': 'frozen-runtime-v6', 'system': a.name, 'family': entry['family'],
        'package_sha256': entry['sha256'], 'package_entry': entry, 'package_entry_sha256': object_sha(entry),
        'plan_sha256_at_launch': sha(a.plan), 'dataset': dataset_name, 'dataset_entry': dataset,
        'cases_sha256': sha(a.cases), 'corpus_sha256': sha(a.kb), 'questions_sha256': sha(data / 'questions.csv'),
        'ids': ids, 'seed': a.seed, 'public_settings': public, 'service_profile_sha256': profile_sha(public),
        'runner_sha256': sha(__file__), 'implementations': implementation_hashes(),
        'historical_psb_config_confirmed': False, 'evaluation_role': entry.get('role', 'candidate'),
        'cache_policy': 'shared exact-text cache; elapsed includes runtime initialization, excludes BGE server warmup',
        'valid_run': False, 'exit_code': None, 'launch_status': 'prepared',
        'embedding_seed_sha256': sha(a.seed_dir / 'bge_embeddings.npz'),
        'embedding_seed_manifest_sha256': sha(a.seed_dir / 'embedding_cache.json')}
    write_json(a.output / 'run.meta.json', meta)
    cache = server = thread = process = None
    stdout = stderr = b''
    begin = time.perf_counter()
    try:
        cache = EmbeddingCache(a.seed_dir / 'bge_embeddings.npz', a.seed_dir / 'embedding_cache.json', a.cache_db, encoder=LazyBGE(a.weights))
        meta['embedding_cache_before'] = cache.stats()
        started = time.perf_counter()
        cache.encoder(['Проверка доступности эмбеддинга.'])
        meta['embedding_warmup_seconds'] = time.perf_counter() - started
        with (a.output / 'embedding.requests.jsonl').open('w', buffering=1) as log:
            server = make_server(cache, log, 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            config.update(EMB_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1', EMB_API_KEY='local-cache', EMB_MODEL=MODEL)
            config.update(EMBEDDER_BASE_URL=config['EMB_BASE_URL'], EMBEDDER_API_KEY=config['EMB_API_KEY'])
            env = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy'}}
            env.update({k: v for k, v in config.items() if not k.startswith(('JUDGE_', 'GENERATOR_'))})
            env.update(CACHE_DIR=str((a.output / 'cache').resolve()), DATA_DIR=str(data.resolve()),
                OUTPUT_CSV=str((a.output / 'submission.csv').resolve()),
                METRICS_PATH=str((a.output / 'submission.metrics.jsonl').resolve()), PIPELINE_MODE='single_pass')
            env['NO_PROXY'] = env.get('NO_PROXY', '') + ',127.0.0.1,localhost'
            env['no_proxy'] = env['NO_PROXY']
            begin = time.perf_counter()
            process = subprocess.Popen([sys.executable, 'solution.py', '--input', str((data / 'questions.csv').resolve()),
                '--output', str((a.output / 'submission.csv').resolve())], cwd=a.output, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            meta['launch_status'] = 'started'
            write_json(a.output / 'run.meta.json', meta)
            try:
                stdout, stderr = process.communicate(timeout=plan.get('outer_seconds_per_system', 1200))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                meta['outer_timeout'] = True
            meta['exit_code'] = process.returncode
            meta['launch_status'] = 'finished'
    except BaseException as exc:
        meta['harness_error_type'] = type(exc).__name__
        meta['launch_status'] = 'harness_error'
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            meta['exit_code'] = process.returncode
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join()
        if cache is not None:
            meta['embedding_cache_after'] = cache.stats()
            cache.close()
    for name, blob in (('stdout.txt', stdout), ('stderr.txt', stderr)):
        text = blob.decode(errors='replace')
        for k, v in config.items():
            if k.endswith('_KEY') and v:
                text = text.replace(v, '[REDACTED]')
        (a.output / name).write_text(text, encoding='utf-8')
    meta['elapsed_seconds'] = round(time.perf_counter() - begin, 3)
    obs, usage, call_count = observe(a.output, entry['family'], ids, entry['files'])
    try:
        answers, issues = read_emitted_answers(a.output / 'submission.csv', ids)
    except (ValueError, OSError):
        answers, issues = {}, {'missing_ids': ids, 'malformed_csv': True}
    meta.update(assessment(meta, obs, answers, issues, ids))
    meta.update(service_observations=obs, token_usage=usage, api_calls_count=call_count,
                observation_sha256=observation_hashes(a.output))
    if (a.output / 'submission.csv').exists():
        meta['answers_sha256'] = sha(a.output / 'submission.csv')
    try:
        verify_runtime(a.output, entry['files'])
    except ValueError:
        meta['valid_run'] = False
        meta['invalid_reasons'].append('runtime_changed_during_execution')
    write_json(a.output / 'run.meta.json', meta)
    print(json.dumps({k: meta[k] for k in ('valid_run', 'invalid_reasons', 'elapsed_seconds', 'service_observations')}, ensure_ascii=False), flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'package', 'cases', 'kb', 'services', 'output', 'seed-dir', 'cache-db', 'weights'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', type=int, default=2026111402)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({'status': 'blocked', 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
    if not result['valid_run']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
