"""Curated, corpus-specific factual hints, enabled only for exact live sources.

Only locally authored hint text can enter the system prompt. Neither questions
nor source text are promoted to system instructions. Unknown/changed sources
continue through ordinary RAG without these static factual hints.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from focused_context import source_name
from token_budget import estimate_tokens
from source_identity import source_matches,live_span


@lru_cache(maxsize=1)
def load_playbook():
    return json.loads(Path(__file__).with_name('corpus_playbook.json').read_text(encoding='utf8'))


def recover_hint_sources(topics,evidence,documents,retrieved_filenames,budget=1800,question='',playbook=None):
    """Recover complete, exact supporting paragraphs from already found docs.

    These are evidence, not instructions. Every note's missing anchors must fit
    together; never truncate a condition or inject a fact absent from live data.
    """
    book=load_playbook() if playbook is None else playbook
    by_filename={d.filename:d for d in documents.values()}
    query=question.lower().replace('ё','е');candidates=[];result=list(evidence);used=0
    for entry in book['documents']:
        d=by_filename.get(entry['filename'])
        if d is None or d.filename not in retrieved_filenames:continue
        if not source_matches(d.raw,entry):continue
        for note in entry['notes']:
            if not set(note['topics']).intersection(topics):continue
            if note.get('when') and not re.search(note['when'],query):continue
            candidates.append(((int(note.get('priority',0)),bool(note.get('when'))),d,note))
    for _,d,note in sorted(candidates,key=lambda item:item[0],reverse=True):
        visible=[e for e in result if e['document']==source_name(d)]
        anchors=[live_span(d.raw,a) for a in note['anchors']]
        if not all(anchors):continue
        missing=list(dict.fromkeys(a[2] for a in anchors if not any(a[2] in e['text'] for e in visible)))
        if not missing:continue
        additions=[]
        for anchor in missing:
            start=d.raw.index(anchor)
            additions.append({'source_id':f'D{len(result)+len(additions)+1}','document':source_name(d),
              'title':getattr(d,'title',source_name(d)),'scope':f'подтверждающее положение, символы {start}:{start+len(anchor)}','text':anchor})
        cost=sum(estimate_tokens(json.dumps(e,ensure_ascii=False)) for e in additions)
        if used+cost>budget:continue
        result.extend(additions);used+=cost
    return result,used


def verified_hints(topics, evidence, documents, budget=2500, playbook=None, question=''):
    book=load_playbook() if playbook is None else playbook
    by_name={source_name(d):d for d in documents.values()}
    active=[];used=0;candidates=[]
    query=question.lower().replace('ё','е')
    for entry in book['documents']:
        doc=next((d for d in by_name.values() if d.filename==entry['filename']),None)
        if doc is None or not source_matches(doc.raw,entry):
            continue
        visible=[e for e in evidence if e['document']==source_name(doc)]
        if not visible:continue
        for note in entry['notes']:
            if not set(note['topics']).intersection(topics):continue
            if note.get('when') and not re.search(note['when'],query):continue
            # Require the actual supporting text in the generator's evidence,
            # not merely a hash of a source that retrieval did not show it.
            refs=[]
            for anchor in note['anchors']:
                span=live_span(doc.raw,anchor)
                match=next((e for e in visible if span and span[2] in e['text']),None)
                if match is None:break
                refs.append(match['source_id'])
            else:
                value={'sources':list(dict.fromkeys(refs)), 'hint':note['hint']}
                # Prefer a precise applicable note before broad background;
                # ordering never changes its source/anchor validation.
                rank=(int(note.get('priority',0)),bool(note.get('when')))
                candidates.append((rank,value))
    for _,value in sorted(candidates,key=lambda item:item[0],reverse=True):
        cost=estimate_tokens(json.dumps(value,ensure_ascii=False))
        if used+cost<=budget:active.append(value);used+=cost
    if not active:return '',0,[]
    prefix=('ПРОВЕРЕННАЯ ПАМЯТКА ПО КОНКРЕТНОЙ БАЗЕ. Это заранее составленные '
            'подсказки разработчика: соответствующие исходные документы сверены '
            'по SHA256 и основания присутствуют в evidence. Используй их для '
            'точного ответа по теме вопроса. Памятка не заменяет условия задачи, '
            'даты применимости и ссылки на исходные положения. Не пересказывай '
            'не относящиеся к вопросу факты из памятки. Если вывод зависит от '
            'нескольких документов, проверь их совместно.\n')
    # Include prompt framing in the same shared context allowance.
    while active:
        text=prefix+json.dumps(active,ensure_ascii=False)
        cost=estimate_tokens(text)
        if cost<=budget:return text,cost,active
        active.pop()
    return '',0,[]
