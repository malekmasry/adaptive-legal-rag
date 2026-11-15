"""Bounded metadata-only diagnostics, visible to the runner as well as on disk."""
import contextvars
import json
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

_request = contextvars.ContextVar('request_id', default='')
_stage = contextvars.ContextVar('stage', default='bootstrap')
_lock = threading.Lock()


@contextmanager
def request_scope(identifier):
    token = _request.set(str(identifier)[:128])
    try:
        yield
    finally:
        _request.reset(token)


@contextmanager
def stage_scope(name):
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)


def error_info(exc):
    # Never stringify arbitrary exceptions: they can contain API keys, payloads,
    # HTTP headers, or entire Pydantic input values.
    frames = traceback.extract_tb(exc.__traceback__)[-6:]
    return {'error_type': type(exc).__name__,
            'error_code': getattr(exc, 'code', type(exc).__name__),
            'frames': [{'file': Path(f.filename).name, 'line': f.lineno,
                        'function': f.name} for f in frames]}


def emit(cache_dir, event, **metadata):
    record = {'event': event, 'request_id': _request.get(), 'stage': _stage.get(),
              'time': round(time.time(), 3), **metadata}
    line = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
    # Logging failure must not replace a valid answer with a technical error.
    with _lock:
        try:
            print(line, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
        try:
            with (Path(cache_dir) / 'diagnostics.jsonl').open('a', encoding='utf8') as f:
                f.write(line + '\n')
        except OSError:
            pass
