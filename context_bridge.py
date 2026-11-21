"""Seed-first residual context expansion; never displace a reranked seed.

Existing cleaned source records and explicit document links only. No inference
of legal precedence, new retrieval, rewriting, or model calls.
"""
from __future__ import annotations
import json
import re
from rag_safety import normalized

REFERENCE = re.compile(r"(?:\bп\.|\bпункт[а-я]*)\s*(\d+(?:\.\d+){0,3})", re.I)
MAX_GROUPS = 8
MAX_PARTS = 4
MAX_GROUP_CHARS = 5000


class Candidates(list):
    def __init__(self, seeds, groups):
        # Preserve normal list semantics for callers; optional groups are metadata.
        super().__init__(seeds)
        self.groups = groups


def _parts(engine, doc, clause):
    return sorted((s for s in engine.by_document.get(doc, []) if s.clause == clause),
                  key=lambda s: s.start)


def _text(engine, source):
    return engine.source_records[source.id]['text']


def _basis_here(engine, source, link):
    basis = normalized(link.basis)
    return bool(basis and any(line and (basis in line or len(line) >= 24 and line in basis)
        for line in map(normalized, _text(engine, source).splitlines())))


def expand_candidates(engine, candidates):
    seeds = list(candidates)
    groups, identities = [], set()
    seed_ids = {s.id for s in seeds}
    anchors = seeds[:6]

    def add(parts, origin, reason):
        parts = [p for p in parts if _text(engine, p).strip()]
        identity = tuple(p.id for p in parts)
        if (not parts or set(identity) <= seed_ids or identity in identities or len(groups) >= 64
                or len(parts) > MAX_PARTS
                or sum(len(_text(engine, p)) for p in parts) > MAX_GROUP_CHARS):
            return
        identities.add(identity)
        groups.append({'parts': parts, 'from': origin.id, 'reason': reason})

    for source in anchors:
        applicable = [link for link in engine.links
                      if link.source_document_id == source.document_id
                      and link.target_clause and _basis_here(engine, source, link)]
        external_clauses = {link.target_clause for link in applicable}
        for link in applicable:
            add(_parts(engine, link.target_document_id, link.target_clause), source,
                'explicit_document_reference')
        for clause in dict.fromkeys(REFERENCE.findall(_text(engine, source))):
            if clause not in external_clauses:
                add(_parts(engine, source.document_id, clause), source, 'local_reference')
        parts = _parts(engine, source.document_id, source.clause)
        if len(parts) <= MAX_PARTS:
            add(parts, source, 'complete_small_clause')
        else:
            position = next((i for i, s in enumerate(parts) if s.id == source.id), None)
            if position is not None:
                add(parts[max(0, position-1):position+2], source, 'adjacent_continuations')

    for link in engine.links:
        if link.relation not in {'amends', 'supplements'}:
            continue
        amendments = sorted(engine.by_document.get(link.source_document_id, []), key=lambda s: s.start)
        amendment_hits = [s for s in anchors if s.document_id == link.source_document_id]
        affected = {x for s in amendments for x in REFERENCE.findall(_text(engine, s))}
        target_hits = [s for s in anchors if s.document_id == link.target_document_id and s.clause in affected]
        # A shared contract identity alone is not enough: require the changed
        # clause among top anchors, or an actual amendment hit from retrieval.
        if amendment_hits or target_hits:
            add(amendments, (amendment_hits or target_hits)[0], 'clause_linked_amendment')
        for source in amendment_hits:
            for clause in dict.fromkeys(REFERENCE.findall(_text(engine, source))):
                add(_parts(engine, link.target_document_id, clause), source, 'amended_original_clause')
    priorities = {'explicit_document_reference': 0, 'local_reference': 1,
                  'clause_linked_amendment': 2, 'amended_original_clause': 3,
                  'complete_small_clause': 4, 'adjacent_continuations': 5}
    groups.sort(key=lambda group: priorities[group['reason']])
    return Candidates(seeds, groups[:MAX_GROUPS])


def pack_candidates(engine, candidates, *, max_chars, max_sources, trace=None):
    """Preserve exactly every seed the ordinary ordered pack can fit.

    Optional groups are atomic, source-grounded, and consume residual budget.
    Existing IDs and payload records are never changed or clipped.
    """
    records, seen, used, skipped, additions = [], set(), 2, [], []

    def key(source, record):
        return source.document_id, normalized(record['text'])

    def fresh(parts):
        pending, new_keys = [], set()
        for source in parts:
            record = engine.source_records[source.id]
            k = key(source, record)
            if not record['text'].strip() or k in seen or k in new_keys:
                continue
            pending.append((source, record, k))
            new_keys.add(k)
        return pending

    def cost(pending):
        return sum(len(json.dumps(r, ensure_ascii=False)) for _, r, _ in pending) + 2 * (
            len(pending) if records else max(0, len(pending)-1))

    def accept(pending):
        nonlocal used
        used += cost(pending)
        for _, record, k in pending:
            records.append(record)
            seen.add(k)

    for source in candidates:
        pending = fresh([source])
        if not pending:
            continue
        if used + cost(pending) > max_chars or len(records) >= max_sources:
            skipped.append(source.id)
            continue
        accept(pending)

    retained_seeds = [r['id'] for r in records]
    for group in getattr(candidates, 'groups', []):
        pending = fresh(group['parts'])
        if not pending:
            continue
        if used + cost(pending) > max_chars or len(records) + len(pending) > max_sources:
            skipped.extend(source.id for source, _, _ in pending)
            continue
        accept(pending)
        for source, _, _ in pending:
            additions.append(source.id)
            if trace:
                trace.events.append({'stage': 'structural_expansion', 'from': group['from'],
                                     'to': source.id, 'reason': group['reason']})
    if trace:
        trace.events.append({'stage': 'context', 'characters': used,
            'source_ids': [r['id'] for r in records], 'skipped_ids': list(dict.fromkeys(skipped)),
            'policy': 'seed_first_residual_atomic', 'retained_seed_ids': retained_seeds,
            'addition_ids': additions})
    return records
