"""Offline parity against the actual donor commit, not an LLM quality score.

Run from this directory. Needs the repository's donor commit and local KB.
No network, credentials or paid APIs. Reproduces both conditional branches.
"""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
from unittest.mock import patch

import numpy as np
from corpus import ingest
from settings import Settings
from schemas import Plan
from retriever35 import Retriever
import retriever35_core as port

ROOT = Path(__file__).resolve().parent
COMMIT = '5d7f10d193edfca14f8c94a7f343e4f70303cc6a'


def main(destination):
    started = time.monotonic()
    git = shutil.which('git')
    preserved = {k:v for k,v in os.environ.items() if k.upper() in ('PATH','SYSTEMROOT','WINDIR','TEMP','TMP')}
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, preserved, clear=True):
        work = Path(directory)
        for name in ('legal_core.py', 'rag_safety.py', 'hallugraph.py', 'rag_telemetry.py', 'single_pass.py'):
            raw = subprocess.check_output([git, 'show', COMMIT + ':' + name], cwd=ROOT)
            (work / name).write_bytes(raw)
        sys.path.insert(0, str(work))
        import legal_core as donor
        import single_pass
        import rag_telemetry
        sys.path.pop(0)
        archive = work / 'kb.zip'
        from corpus import read_sources
        with zipfile.ZipFile(archive, 'w') as z:
            for name, raw in read_sources(ROOT / 'data'):
                z.writestr(name, raw)
        settings = Settings(data_dir=ROOT / 'data', cache_dir=work / 'cache',
                            base_url='https://offline.invalid', api_key='offline-test')
        docs, chunks, relations = ingest(settings.data_dir)

        class OfflineAPI:
            route = 'high'
            calls = []
            def post(self, path, payload, kind):
                self.calls.append(kind)
                assert path == '/rerank'
                count = len(payload['documents'])
                order = list(range(count))
                order = order[2:] + order[:2]
                score = .9 if self.route == 'high' else .1
                if self.route == 'error':
                    raise RuntimeError('offline reranker failure')
                if self.route == 'partial':
                    order = order[:3]
                return {'results': [{'index': i, 'relevance_score': score / (j+1)} for j, i in enumerate(order)]}
            def embed(self, texts):
                self.calls.append('embedding')
                # Deterministic protocol-test vectors, NOT relevance evaluation.
                values = []
                for text in texts:
                    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))
                    value = rng.standard_normal(1024)
                    values.append(value / np.linalg.norm(value))
                return np.asarray(values)

        api = OfflineAPI()
        actual = Retriever(docs, chunks, relations, api, settings)
        expected = donor.LegalAssistant(archive, actual.engine.proxy, use_embeddings=False)
        expected.retrieval_mode = 'conditional'
        assert len(expected.sources) == len(actual.chunks) == 342
        assert expected.search_texts == actual.engine.search_texts
        assert expected.source_records == actual.engine.source_records
        assert actual.vectors is not None, 'Bundle the genuine document embeddings before running parity'
        expected.vectors = actual.vectors.tolist()
        # Verify exact function bodies for the transferred ranking algorithms.
        donor_ast = ast.parse((work / 'legal_core.py').read_text(encoding='utf8'))
        port_ast = ast.parse((ROOT / 'retriever35_core.py').read_text(encoding='utf8'))
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LegalAssistant')
            return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
        a, b = methods(donor_ast), methods(port_ast)
        for name in b:
            if name != '__init__':
                assert a[name] == b[name], name
        cases = json.loads((ROOT / 'evaluation_cases.json').read_text(encoding='utf8'))
        rows = []
        for route in ('high', 'low', 'partial', 'error'):
            api.route = route
            for case in cases if route == 'high' else cases[:4]:
                api.calls = []
                query = port.clean_data(case['question'])[0]
                trace = rag_telemetry.Trace()
                token = rag_telemetry.CURRENT.set(trace)
                try:
                    expected_records = single_pass.pack_context(expected, single_pass.expand_context(expected, expected.retrieve(query)))
                finally:
                    rag_telemetry.CURRENT.reset(token)
                expected_calls = api.calls[:]
                api.calls = []
                result, diagnostics = actual.search(Plan(question=case['question']))
                ids = [c.id for c in result]
                assert ids == [r['id'] for r in expected_records], (route, case['id'])
                assert api.calls == expected_calls, (route, case['id'], api.calls, expected_calls)
                if route == 'low':
                    assert diagnostics['route']['route'] == 'dense'
                if route == 'high':
                    assert diagnostics['route']['status'] == 'lexical_confident'
                rows.append({'id': case['id'], 'route': route, 'selected': ids, 'calls': api.calls[:]})
        report = {'donor_commit': COMMIT, 'documents': len(docs), 'chunks': len(actual.chunks),
                  'search_texts_identical': True, 'ranking_methods_ast_identical': True,
                  'parity_cases': len(rows), 'all_passed': True, 'paid_calls': 0,
                  'scope': 'execution and donor parity; scripted reranker, not an LLM quality score',
                  'seconds': round(time.monotonic()-started, 3), 'cases': rows}
        Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1])
