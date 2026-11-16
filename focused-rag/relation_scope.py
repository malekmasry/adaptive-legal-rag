"""Question-aware expansion of source relationships; no source is edited.

Primary reranker hits are never removed here. Only automatic relationship
expansion is gated. Unrecognized questions/rules and changed sources keep the
original relationship. Temporal conditions remain visible, not guessed filters.
"""
import hashlib
import json
from pathlib import Path
from functools import lru_cache
import re
from answer_guides import plan_for,numbered_span
from source_identity import source_matches


@lru_cache(maxsize=1)
def rules():
    return json.loads(Path(__file__).with_name('relation_scope.json').read_text(encoding='utf8'))['relations']


def question_topics(question):
    q=question.lower().replace('ё','е');topics=set(plan_for(q)['topics'])
    if re.search(r'согласован|согласовать|согласующ',q):topics.add('approval')
    if re.search(r'иностран|валютн|санкцион|комплаенс|к-12',q):topics.add('foreign_checks')
    if re.search(r'адрес|куда|везти|склад\w*\s+достав',q):topics.add('delivery_address')
    if re.search(r'доп\.?\s*соглаш|дополнительн\w*\s+соглаш',q):topics.add('amendment_question')
    if re.search(r'ндс|налог\w*\s+на\s+добавлен|ст\.?\s*(?:145|164)',q):topics.add('vat')
    broad=bool(re.search(r'все\s+(?:измен|услов|требован)|в\s+целом|что\s+измен|перечисл\w*\s+измен',q))
    return topics,broad


def decide(relation,documents,question,table=None):
    source=documents[relation['source']];target=documents[relation['target']]
    table=rules() if table is None else table
    rule=next((r for r in table if r['source']==source.filename and r['target']==target.filename
               and r['kind']==relation['kind']),None)
    if rule is None:return True,None,'unknown_relation_keep'
    for side,doc in [('source',source),('target',target)]:
        if not source_matches(doc.raw,{'sha256':rule[side+'_sha256'],
                                      'canonical_sha256':rule.get(side+'_canonical_sha256')}):
            return True,None,'changed_source_keep'
    topics,broad=question_topics(question)
    if not topics or broad:return True,None,'uncertain_question_keep'
    relevant=bool(topics.intersection(rule['topics']))
    return relevant,rule,('relevant' if relevant else 'unrelated_topic_skip')


def project(document,selectors):
    """Only complete verbatim source provisions; no compressed paraphrase."""
    if not selectors:return None
    spans=[]
    for selector in selectors:
        if selector.startswith('line:'):
            matches=[x for x in document.raw.splitlines() if x.startswith(selector[5:])]
            if len(matches)!=1:return None
            text=matches[0]
        else:text=numbered_span(document.raw,selector)
        if not text or text not in document.raw:return None
        if text not in spans:spans.append(text)
    return spans


def expand_relations(primary,documents,relations,question,table=None):
    order=list(primary);selected=set(primary);projections={};decisions=[]
    for rel in relations:
        if rel['kind'] not in ('amends','supplements','supersedes','cites'):continue
        if not ({rel['source'],rel['target']} & selected):continue
        keep,rule,reason=decide(rel,documents,question,table)
        decisions.append({'source':documents[rel['source']].filename,'target':documents[rel['target']].filename,
            'kind':rel['kind'],'keep':keep,'reason':reason,
            'affected_provisions':rule.get('affected_provisions',[]) if rule else [],
            'applicability':rule.get('applicability',{}) if rule else {}})
        if not keep:continue
        for side in ('source','target'):
            identifier=rel[side]
            if identifier in selected:continue
            if identifier not in order:order.append(identifier)
            selectors=rule.get(side+'_selectors') if rule else None
            spans=project(documents[identifier],selectors)
            if spans is None:projections[identifier]=None
            elif identifier not in projections:projections[identifier]=spans
            elif projections[identifier] is not None:
                projections[identifier]=list(dict.fromkeys(projections[identifier]+spans))
    return order,projections,decisions
