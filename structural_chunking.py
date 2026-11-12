"""Dense-only, paragraph-preserving legal passages with source-backed context.

No LLM summaries, corpus-specific answers, or third-party tokenizer. The size is
a soft character target: an indivisible paragraph is never cut to meet it.
Offsets always refer to the unmodified input document.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from rag_safety import clean_data

ARTICLE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s+", re.I)
NUMBER = re.compile(r"^(\d+(?:[.\-]\d+)*)([.)])\s+")
NOTE = re.compile(
    r"^(?:Федеральным(?:и)? закон(?:ом|ами)|"
    r"(?:Статья|Часть|Пункт|Подпункт|Абзац|Абзацы|Примечания)\b.*?"
    r"(?:изменен|изменён|изменена|изменены|дополнен|вступает в силу)|"
    r"(?:О |Об |Согласно ).*(?:см\.|Постановлен|постановлен|Определен)|"
    r"(?:Действие положений|Положения (?:подпункта|пункта|абзаца))\b)", re.I)


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Passage:
    document_id: str
    clause: str
    start: int
    end: int
    context: tuple[Span, ...]
    provisions: tuple[str, ...]


def lines(text: str):
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            leading = len(line) - len(line.lstrip())
            yield Span(offset + leading, offset + len(line.rstrip())), stripped
        offset += len(line)


def passages(document, target: int = 1400) -> list[Passage]:
    """Retain whole paragraphs and scope for lists, parts and subparagraphs.

    Dot-separated law parts are peers (e.g. 1.1 and 1.2), not automatically
    children of part 1. A ')' marker denotes a subparagraph. Repeated numbers
    in a notes section have a separate namespace.
    """
    text = document.text
    rows = list(lines(text))
    if not rows:
        return []
    result = []
    article = top = sub = intro = section = None
    article_name, top_name, sub_name, section_name = "", "", "", ""
    namespace = ""
    pending = None
    atoms = []
    group = None

    def flush():
        if not atoms:
            return
        start, end = atoms[0][0], atoms[-1][1]
        contexts = sorted({s for atom in atoms for s in atom[2]
                           if not (start <= s.start and s.end <= end)}, key=lambda s: (s.start, s.end))
        # A heading containing an earlier note can subsume the same note span.
        contexts = [s for s in contexts if not any(
            other != s and other.start <= s.start and s.end <= other.end for other in contexts)]
        keys = tuple(dict.fromkeys(key for atom in atoms for key in atom[3]))
        result.append(Passage(document.id, atoms[0][4], start, end, tuple(contexts), keys))
        atoms.clear()

    for span, value in rows:
        heading = ARTICLE.match(value) if document.kind == "law" else None
        marker = NUMBER.match(value)
        is_section = (document.kind == "company" and marker and marker[2] == "."
                      and "." not in marker[1] and "-" not in marker[1]
                      and len(value) < 100 and not value.endswith((".", ";", ":")))
        # Structural headings travel with their first substantive paragraph.
        if heading or is_section or value.casefold() == "примечания:":
            flush()
            if pending is None:
                pending = span.start
            if heading:
                article, article_name = span, "статья " + heading[1]
                top = sub = intro = None
                top_name = sub_name = namespace = ""
            elif is_section:
                section, section_name = span, marker[1]
            else:
                namespace = "примечания"
                top = sub = intro = None
                top_name = sub_name = ""
                section = span
            continue
        if document.kind == "law" and NOTE.match(value) and not marker:
            if pending is None:
                pending = span.start
            continue

        start = pending if pending is not None else span.start
        pending = None
        scope_start = start
        # Document-level amendment notices preceding the actual article title
        # must not become the inherited meaning of a later list introduction.
        # Notices after that title may describe this particular part.
        if article is not None and start <= article.start < span.start:
            scope_start = article.end
            while scope_start < span.start and text[scope_start].isspace():
                scope_start += 1
        attached = Span(scope_start, span.end)
        if document.kind == "law":
            if marker and marker[2] == ".":
                top, top_name = attached, marker[1]
                sub = intro = None
                sub_name = ""
            elif marker and marker[2] == ")":
                sub, sub_name = attached, marker[1]
                # A top-level unnumbered list introduction (e.g. consent)
                # applies to every numbered item of that list.
                if top is not None:
                    intro = None
            elif intro is not None and value[0].isupper():
                intro = None
                if top is None:
                    sub, sub_name = None, ""
            # A rule explicitly referring to this entire part is broader than
            # the preceding subparagraph. Do not attach a revoked subparagraph
            # as context for e.g. the common product-code rule in a tax list.
            if not marker and sub and re.search(r"настоящ(?:его|ем|ему|ий) пункт", value, re.I):
                sub, sub_name = None, ""
            if value.endswith(":"):
                if top is None and not marker:
                    sub, sub_name = None, ""
                intro = attached
            clause = " / ".join(x for x in (article_name, namespace,
                      "пункт " + top_name if top_name else "",
                      "подпункт " + sub_name if sub_name else "") if x) or "текст"
            scopes = tuple(s for s in (article, section, top, sub, intro) if s is not None)
            key = (clause, intro.start if intro else None)
            provisions = (top_name,) if top_name else ()
        else:
            clause = section_name or (marker[1] if marker else "текст")
            scopes = (section,) if section else ()
            key = (clause,)
            provisions = tuple(x for x in (section_name, marker[1] if marker else "") if x)
        if atoms and (key != group or span.end - atoms[0][0] > target):
            flush()
        group = key
        atoms.append((start, span.end, scopes, provisions, clause))
    # A trailing editorial note remains available, but is never treated as an
    # article heading or used to invent the missing text of an added part.
    if pending is not None:
        if atoms:
            last = atoms[-1]
            atoms[-1] = (last[0], rows[-1][0].end, *last[2:])
        elif result:
            last = result[-1]
            result[-1] = Passage(last.document_id, last.clause, last.start,
                                 rows[-1][0].end, last.context, last.provisions)
        else:
            atoms.append((rows[0][0].start, rows[-1][0].end, (), (), "текст"))
    flush()
    return result


def build_dense_sources(documents, legacy_sources, source_type):
    """Return citable originals, search records and paragraph provenance.

    New IDs live after the unchanged lexical IDs; all query-time state is local.
    The same enriched record is used by embedding, dense reranking and answering.
    """
    from hallugraph import revision_metadata
    sources, records = [], {}
    legacy_by_doc = {}
    for source in legacy_sources:
        legacy_by_doc.setdefault(source.document_id, []).append(source)
    for doc in documents:
        body, _ = clean_data(doc.text)
        revision = revision_metadata(body)
        # Preserve source-backed document dates in their original wording. A
        # snapshot date or notice of a future amendment is NOT an effective date.
        document_context = []
        for span, value in lines(doc.text):
            if ARTICLE.match(value) or NUMBER.match(value):
                break
            if (re.search(r"редакци|действует с|утратил[ао]? силу", value, re.I)
                    and not NOTE.match(value)):
                document_context.append(span)
        for unit in passages(doc):
            sid = f"S{len(legacy_sources) + len(sources) + 1:04d}"
            raw = doc.text[unit.start:unit.end]
            source = source_type(sid, doc.id, doc.title, doc.path, unit.clause, raw, unit.start, unit.end)
            sources.append(source)
            spans = sorted(set(unit.context + tuple(document_context)), key=lambda s: (s.start, s.end))
            spans = [s for s in spans if not (unit.start <= s.start and s.end <= unit.end)]
            parent = "\n".join(clean_data(doc.text[s.start:s.end])[0] for s in spans)
            parties = " ".join(f"«{a or b}»" for a, b in re.findall(r'«([^»]+)»|"([^"\n]+)"', body[:3000])[:10])
            records[sid] = dict(id=sid, document_id=doc.id, document=clean_data(doc.title)[0],
                path=clean_data(doc.path)[0], kind=doc.kind, clause=unit.clause, text=clean_data(raw)[0],
                parent_scope=parent, contract_number=doc.contract_number, parties=parties, revision=revision,
                source_start=unit.start, source_end=unit.end,
                context_offsets=[dict(start=s.start, end=s.end) for s in spans],
                provisions=list(unit.provisions),
                original_source_ids=[s.id for s in legacy_by_doc.get(doc.id, ())
                                     if s.start < unit.end and unit.start < s.end])
    return sources, records
