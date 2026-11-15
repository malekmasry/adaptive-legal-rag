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


@lru_cache(maxsize=1)
def load_playbook():
    return json.loads(Path(__file__).with_name('corpus_playbook.json').read_text(encoding='utf8'))


def verified_hints(topics, evidence, documents, budget=2500, playbook=None, question=''):
    book=load_playbook() if playbook is None else playbook
    by_name={source_name(d):d for d in documents.values()}
    active=[];used=0;candidates=[]
    query=question.lower().replace('ё','е')
    for entry in book['documents']:
        doc=next((d for d in by_name.values() if d.filename==entry['filename']),None)
        if doc is None or hashlib.sha256(doc.raw.encode('utf8')).hexdigest()!=entry['sha256']:
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
                match=next((e for e in visible if anchor in e['text']),None)
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
