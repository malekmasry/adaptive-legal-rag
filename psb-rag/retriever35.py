"""Donor retrieval adapted to v8 evidence and transport contracts.

Only retrieval changes: v8 still owns plans, generation, verification, security,
session memory, API retries and the global deadline. No generator from the donor.
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from corpus import Chunk, read_sources
from model_api import ModelError
from retriever35_core import (Document, CONTRACT_RE, LegalAssistant, CURRENT,
                             clean_data, expand_context, pack_context)


class Proxy:
    def __init__(self, api, settings):
        self.api = api
        self.base, self.key = settings.base_url, settings.api_key
        self.embed_base, self.embed_key = settings.emb_url, settings.emb_key
        self.rerank_base, self.rerank_key = settings.rerank_url, settings.rerank_key
        self.rerank_model = settings.rerank_model if settings.use_rerank else ''

    def embeddings(self, texts):
        if self.embed_base and not self.embed_key:
            raise ModelError('Separate embedding credential is missing')
        return self.api.embed(texts).tolist()

    def rerank_with_scores(self, question, passages):
        # Donor used the question verbatim and requested scores for the WHOLE pool.
        # Top-6 truncation or a new instruction would change its confidence gate.
        if self.rerank_base and not self.rerank_key:
            raise ModelError('Separate reranker credential is missing')
        data = self.api.post('/rerank', {
            'model': self.rerank_model, 'query': question, 'documents': passages,
            'top_n': len(passages),
        }, 'rerank')
        ranked = data.get('results', data.get('data', []))
        if not isinstance(ranked, list) or not all(isinstance(item, dict) for item in ranked):
            raise ValueError('Invalid reranker response')
        result, seen = [], set()
        for item in ranked:
            index = item.get('index')
            if type(index) is not int or not 0 <= index < len(passages) or index in seen:
                continue
            seen.add(index)
            score = item.get('relevance_score', item.get('score'))
            score = float(score) if type(score) in (int, float) and math.isfinite(score) else None
            result.append((index, score))
        return result


class Retriever:
    def __init__(self, docs, chunks, relations, api, settings):
        self.s, self.api = settings, api
        self.docs = {d.id: d for d in docs}
        by_name = {d.filename: d for d in docs}
        documents, mapping = [], {}
        for path, raw in sorted(read_sources(settings.data_dir)):
            filename = path.replace('\\', '/').rsplit('/', 1)[-1]
            d = by_name[filename]
            identifier = hashlib.sha1(path.encode('utf8')).hexdigest()[:12]
            contract = CONTRACT_RE.search(raw[:600])
            documents.append(Document(identifier, path, raw.strip().splitlines()[0].strip()[:200], raw,
                'law' if re.search(r'(?m)^\s*Статья\s+\d+', raw) else 'company',
                contract.group(1).upper() if contract else None))
            mapping[identifier] = d.id
        self.engine = LegalAssistant(documents, Proxy(api, settings), use_embeddings=False)
        self.audit_chunks = chunks  # References updated by v8 classify(), if needed.
        self.chunks = []
        for ordinal, source in enumerate(self.engine.sources):
            doc_id = mapping[source.document_id]
            locator = ('п. ' + source.clause if re.fullmatch(r'\d+(?:\.\d+)*', source.clause)
                       else source.clause)
            notes = []
            for original in chunks:
                if original.doc_id == doc_id and original.start < source.end and original.end > source.start:
                    for note in original.annotations:
                        if note not in notes:
                            notes.append(note)
            self.chunks.append(Chunk(source.id, doc_id, locator, source.start, source.end,
                source.text, '', ordinal, 'passage', notes,
                prefix=self.engine.source_records[source.id]['parent_scope']))
        self.all = {c.id: c for c in self.chunks}
        self.text_hashes = [hashlib.sha256(t.encode('utf8')).hexdigest() for t in self.engine.search_texts]
        self.cache_key = hashlib.sha256(json.dumps({
            'retriever': 'conditional35-5d7f10d', 'model': settings.emb_model,
            'texts': self.text_hashes}, sort_keys=True).encode()).hexdigest()
        self.cache_path = settings.cache_dir / ('vectors35-' + self.cache_key + '.npz')
        self.vectors = None
        for path in (self.cache_path, Path(__file__).with_name(self.cache_path.name)):
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as data:
                        vectors = data['vectors']
                        if list(data['text_hashes']) != self.text_hashes:
                            raise ValueError('Embedding text mismatch')
                        self._check_vectors(vectors)
                        self._set_vectors(vectors)
                        break
                except (ValueError, KeyError, OSError):
                    continue

    def _check_vectors(self, vectors):
        if (vectors.ndim != 2 or len(vectors) != len(self.chunks) or not vectors.shape[1]
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid donor embedding index')

    def _set_vectors(self, vectors):
        self.vectors = vectors
        self.engine.vectors = vectors.tolist()

    def prepare_dense(self):
        if self.vectors is not None or self.s.retrieval_mode == 'bm25':
            return
        # Just as in the donor, dense retrieval requires confidence scores.
        if not self.engine._reranker_ready():
            return
        vectors = np.asarray(self.engine.proxy.embeddings(self.engine.search_texts), dtype=np.float32)
        try:
            self._check_vectors(vectors)
        except ValueError as exc:
            raise ModelError(str(exc)) from None
        temp = self.cache_path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, vectors=vectors, text_hashes=np.array(self.text_hashes))
        temp.replace(self.cache_path)
        self._set_vectors(vectors)

    def search(self, plan, mode=None, original_query=''):
        started = time.monotonic()
        trace = SimpleNamespace(stage='retrieval', events=[], timings={})
        token = CURRENT.set(trace)
        try:
            # v8 resolves multi-turn questions. No v8 fusion, date/entity pool
            # filtering, MMR or second rerank may alter the donor's selected list.
            query, _ = clean_data(plan.question or original_query)
            if self.s.retrieval_mode == 'bm25' or mode == 'bm25':
                from retriever35_core import focused_search_question
                lexical = self.engine._lexical(focused_search_question(query))
                candidates = self.engine._expand_retrieval(lexical, lexical, self.s.top_k)
            else:
                candidates = self.engine.retrieve(query, limit=self.s.top_k)
            records = pack_context(self.engine, expand_context(self.engine, candidates))
            selected = [self.all[r['id']] for r in records]
            # Keep raw evidence, offsets and v8 security annotations. Donor
            # sanitation affects search representations, never the connected KB.
            # Return fresh objects so concurrent requests cannot mutate each other.
            from dataclasses import replace
            result = []
            for chunk in selected:
                spans = []
                for old in self.audit_chunks:
                    if old.doc_id == chunk.doc_id:
                        for span in old.instruction_spans:
                            if span['start'] < chunk.end and span['end'] > chunk.start and span not in spans:
                                spans.append(span)
                result.append(replace(chunk, instruction_spans=spans))
            route = next((e for e in trace.events if e['stage'] == 'retrieval_route'), {})
            return result, {'mode': 'conditional35' if self.s.retrieval_mode != 'bm25' else 'bm25',
                'donor_commit': '5d7f10d', 'pool': len(self.chunks), 'selected': [c.id for c in result],
                'route': route, 'events': trace.events, 'timings_ms': trace.timings,
                'seconds': round(time.monotonic() - started, 4), 'degraded': []}
        finally:
            CURRENT.reset(token)
