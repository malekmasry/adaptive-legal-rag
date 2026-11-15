"""Retrieve passages, then restore the source scope needed to interpret them."""
import json
import re
from retrieval import Retriever
from token_budget import estimate_tokens


def search_queries(question):
    # Earlier assistant messages are unverified claims, not retrieval facts.
    turns = re.split(r'(?im)^\s*(Пользователь|User|Ассистент|Assistant)\s*:\s*', question)
    users = [turns[i+1].strip() for i in range(1, len(turns)-1, 2)
             if turns[i].lower() in ('пользователь', 'user')]
    if not users:
        return question, []
    latest = users[-1]
    combined = '\n'.join(users[-4:])
    return combined, [latest] if latest != combined else []


class ScopeRetriever(Retriever):
    def pool(self, plan):
        # Dates and similar names are decided against actual source text. A
        # guessed router must not discard an amendment or the historical edition.
        return list(range(len(self.chunks)))


def source_name(document):
    return document.filename.removesuffix('.txt').replace('документы компании__', '').replace('законы__', '')


def context_bundle(chunks, documents, relations, budget):
    """Full small documents; bounded, verbatim windows for large articles.

    Full text includes suspicious instructions unchanged. Only the system
    message determines authority. No question-specific answers live here.
    """
    order = list(dict.fromkeys(c.doc_id for c in chunks))
    # Show linked revisions and amendments together, including dates and scope.
    # Do not recursively pull the entire legal corpus through citation chains.
    selected = set(order)
    for relation in relations:
        if relation['kind'] in ('amends', 'supplements', 'supersedes'):
            if relation['source'] in selected and relation['target'] not in order:
                order.append(relation['target'])
            if relation['target'] in selected and relation['source'] not in order:
                order.append(relation['source'])
    candidates = []
    for doc_id in order:
        d = documents[doc_id]
        if len(d.raw) <= 14000:
            candidates.append((d, 'полный документ', d.raw))
        else:
            spans = sorted({(c.start, c.end) for c in chunks if c.doc_id == doc_id})
            merged = []
            for start, end in spans:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            # Header identifies the article; bodies retain all list qualifiers.
            for start, end in merged:
                candidates.append((d, f'фрагмент, символы {start}:{end}', d.raw[start:end]))
    evidence, used, seen, omitted = [], 0, set(), []
    # Count the serialized payload, rather than estimating text alone.
    for d, scope, text in candidates:
        if (d.id, text) in seen:
            continue
        item = {'source_id':f'D{len(evidence)+1}', 'document':source_name(d),
                'title':d.title, 'scope':scope, 'text':text}
        cost = estimate_tokens(json.dumps(item, ensure_ascii=False))
        if used + cost > budget:
            omitted.append(d.filename)
            continue
        evidence.append(item); used += cost; seen.add((d.id, text))
    return evidence, {'estimated_tokens':used, 'budget':budget,
                      'omitted_documents':list(dict.fromkeys(omitted)), 'full_documents':sum(e['scope']=='полный документ' for e in evidence)}


def render_paragraphs(paragraphs, evidence):
    sources = {e['source_id']:e['document'] for e in evidence}
    lines, refs = [], []
    for paragraph in paragraphs:
        text = paragraph['text'].strip()
        citations = {}
        for reference in paragraph['citations']:
            identifier, locator = reference['source_id'], reference['locator'].strip()
            if identifier not in sources:
                raise ValueError('Unknown source reference: '+identifier)
            if not locator:
                raise ValueError('Missing provision locator')
            citations.setdefault(sources[identifier],[]).append(locator)
            refs.append(identifier)
        # Source ID and provision are separate typed fields, not inline markup
        # that can fail merely because the model chooses different punctuation.
        labels=[document+', '+', '.join(dict.fromkeys(locators)) for document,locators in citations.items()]
        lines.append(text + (' ('+'; '.join(labels)+')' if labels else ''))
    answer='\n\n'.join(lines)
    if not answer or len(answer)>10000:
        raise ValueError('Answer length outside platform contract')
    return answer,refs


def date_note(question, answer, as_of, has_sources):
    if not has_sources:
        return answer
    query,_=search_queries(question)
    # Bare years in contract identifiers are not event dates. Explicit calendar
    # and relative dates remain the generator's responsibility; do not label a
    # historical answer as current merely because its source was retrieved now.
    has_date=re.search(r'\b\d{1,2}[./]\d{1,2}[./]20\d{2}\b|\b20\d{2}-\d{2}-\d{2}\b|'
        r'\b20\d{2}\s*(?:г(?:од(?:а|у|ом|е)?|\.)|[-–]го)(?!\w)|\b(?:в|за|на)\s+20\d{2}\b|'
        r'(?:январ|феврал|март|апрел|ма[йея]|июн|июл|август|сентябр|октябр|ноябр|декабр)[^\n]{0,24}20\d{2}|'
        r'\b(?:вчера|позавчера|завтра|послезавтра|прошл\w+\s+(?:год|месяц))\b',query,re.I)
    if has_date:
        return answer
    y,m,d=as_of.split('-')
    note=f'По состоянию на {d}.{m}.{y}.\n\n'
    if len(note+answer)>10000:
        raise ValueError('Answer length outside platform contract')
    return note+answer
