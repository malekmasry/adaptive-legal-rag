"""Bounded structural evidence completion using existing, cleaned source records.

The bridge changes context construction only. It never scores retrieval results,
creates citations, infers legal precedence, or reads raw text into model context.
"""
from __future__ import annotations

import re

from rag_safety import normalized


MAX_CLAUSE_PARTS = 6
MAX_CLAUSE_CHARS = 6000
MAX_AMENDMENT_PARTS = 4
MAX_AMENDMENT_CHARS = 4500
MAX_ADDITIONS = 8
REFERENCE = re.compile(r"(?:\bп\.|\bпункт[а-я]*)\s*(\d+(?:\.\d+){0,3})", re.I)


def clause_parts(engine, source):
    """Existing fragments only, isolated by document identity and exact locator."""
    return sorted((item for item in engine.by_document[source.document_id]
                   if item.clause == source.clause), key=lambda item: item.start)


def small_group(engine, group, max_parts=MAX_CLAUSE_PARTS, max_chars=MAX_CLAUSE_CHARS):
    return (len(group) <= max_parts
            and sum(len(engine.source_records[item.id]["text"]) for item in group) <= max_chars)


def basis_overlaps(engine, source, link):
    """A document-level link applies here only where its evidence line occurs."""
    basis = normalized(link.basis)
    lines = [normalized(line) for line in engine.source_records[source.id]["text"].splitlines()]
    return bool(basis and any(text and (basis in text or (len(text) >= 24 and text in basis))
                              for text in lines))


def expand_candidates(engine, candidates, trace=None):
    if not candidates:
        return []
    # Keep the first four retrieval hits in place. Explicit evidence can promote
    # an already retrieved tail hit; it need not be a new hit to need context.
    head = candidates[:4]
    extra, seen = [], {source.id for source in head}
    all_candidates = {source.id for source in candidates}
    additions = 0

    def add(group, origin, reason, atomic=False):
        nonlocal additions
        group = [item for item in group if item.id not in seen
                 and engine.source_records[item.id]["text"].strip()]
        cost = sum(item.id not in all_candidates for item in group)
        if atomic and additions + cost > MAX_ADDITIONS:
            return False
        for item in group:
            if item.id not in all_candidates and additions >= MAX_ADDITIONS:
                break
            seen.add(item.id)
            extra.append(item)
            additions += item.id not in all_candidates
            if trace:
                trace.events.append({"stage": "structural_expansion", "from": origin.id,
                                     "to": item.id, "reason": reason})
        return True

    for source in candidates[:6]:
        parts = clause_parts(engine, source)
        if small_group(engine, parts):
            add(parts, source, "complete_clause", atomic=True)
        else:
            # For a large clause preserve the governing start and nearby
            # fragments without reconstructing or dumping the whole article.
            position = next(i for i, item in enumerate(parts) if item.id == source.id)
            neighbors = [parts[0]] + parts[max(0, position - 1):position + 2]
            add(list({item.id: item for item in neighbors}.values()), source, "partial_clause")

        text = engine.source_records[source.id]["text"]
        references = set(REFERENCE.findall(text))
        applicable = [link for link in engine.links
                      if link.source_document_id == source.document_id
                      and link.target_clause and basis_overlaps(engine, source, link)]
        external = {link.target_clause for link in applicable}
        for link in applicable:
            target = sorted((item for item in engine.by_document[link.target_document_id]
                             if item.clause == link.target_clause), key=lambda item: item.start)
            if small_group(engine, target):
                add(target, source, "explicit_document_reference", atomic=True)
        local = sorted((item for item in engine.by_document[source.document_id]
                        if item.clause in references - external), key=lambda item: item.start)
        # Group each local clause atomically when small, so its exception is
        # not dropped merely because its first fragment was added first.
        for clause in dict.fromkeys(item.clause for item in local):
            group = [item for item in local if item.clause == clause]
            if small_group(engine, group):
                add(group, source, "local_reference", atomic=True)

    seeds = candidates[:6]
    for link in engine.links:
        if link.relation not in {"amends", "supplements"}:
            continue
        involved = [source for source in seeds if source.document_id in
                    {link.source_document_id, link.target_document_id}]
        if not involved:
            continue
        amendment = sorted(engine.by_document[link.source_document_id], key=lambda item: item.start)
        if small_group(engine, amendment, MAX_AMENDMENT_PARTS, MAX_AMENDMENT_CHARS):
            # Complete small amending instruments include their commencement
            # and applicability clauses, not just the changed numeric rule.
            add(amendment, involved[0], "linked_amendment", atomic=True)
        else:
            target_clauses = {item.clause for item in seeds if item.document_id == link.target_document_id}
            for item in amendment:
                if target_clauses.intersection(REFERENCE.findall(engine.source_records[item.id]["text"])):
                    group = clause_parts(engine, item)
                    if small_group(engine, group):
                        add(group, involved[0], "targeted_amendment", atomic=True)
        # When the amendment itself is retrieved, retain its explicitly named
        # original clause too. A relation is evidence to inspect, not precedence.
        amendment_hits = [item for item in seeds if item.document_id == link.source_document_id]
        references = {value for item in amendment_hits
                      for value in REFERENCE.findall(engine.source_records[item.id]["text"])}
        for clause in sorted(references):
            target = sorted((item for item in engine.by_document[link.target_document_id]
                             if item.clause == clause), key=lambda item: item.start)
            if small_group(engine, target):
                add(target, involved[0], "amended_clause", atomic=True)
    return list({source.id: source for source in head + extra + candidates[4:]}.values())


def prepared_record(engine, source, parts):
    """Retain original IDs and text; make split-clause coverage observable."""
    record = dict(engine.source_records[source.id])
    if len(parts) > 1:
        record["clause_fragment"] = {"index": next(i for i, item in enumerate(parts, 1)
                                                  if item.id == source.id), "total": len(parts)}
    return record
