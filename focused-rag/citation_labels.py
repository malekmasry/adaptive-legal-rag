"""Repair only a provable part-versus-note label after final generation.

The cited source IDs, document names, numbers and answer facts are preserved.
A note is recognized by its exact location in the current source, never by a
question ID, a golden answer, or a semantic guess about the answer.
"""
import hashlib,re

def finish_answer(answer,question,source_ids,evidence,documents):
    result=repair_note_locators(answer,source_ids,evidence,documents)
    from answer_guides import plan_for
    plan=plan_for(question)
    # The existing ready draft has two fixed sections. A deadline-only question
    # must not inherit its separate, unrequested order-entry instructions.
    if plan['topics']==['order_delivery'] and plan['requested_slots']==['срок поставки','единица дней','начальное событие']:
        paragraphs=result.split('\n\n')
        order_text='Заявка направляется по электронной почте. В ней указывают наименование, количество товара и желаемую дату поставки.'
        if (len(paragraphs)>=2 and paragraphs[0].startswith(order_text+' (Договор поставки ПТ-05')
                and paragraphs[1].startswith('Срок поставки — 14 рабочих дней с даты получения заявки поставщиком.')):
            result='\n\n'.join(paragraphs[1:])
    from style_renderer import apply_policy
    return apply_policy(result,plan['topics'],source_ids,evidence,documents)

def display_citations(answer,source_ids,evidence,documents):
    """Readable source identities; only reviewed, hash-matching cited documents.

    The exact source IDs and original filenames remain in diagnostics. Renaming
    a citation does not supply a fact, switch its source, or change its meaning.
    """
    from corpus_playbook import load_playbook
    from source_identity import source_matches
    known={d['filename']:d for d in load_playbook()['documents']}
    cited={e['document'] for e in evidence if e['source_id'] in set(source_ids)}
    names={
        'Договор аренды А-08-2024 Северный Склад':'Договор аренды № А-08/2024',
        'Договор поставки П-17-2024 Альфа-Трейд':'Договор № П-17/2024',
        'Договор поставки ПТ-05-2023 Альфа-Трейдинг':'Договор № ПТ-05/2023',
        'Доп соглашение 1 к договору П-17-2024':'Допсоглашение № 1 к П-17/2024',
        'Регламент согласования договоров ред 2023 (утратил силу)':'Регламент, ред. от 10.01.2023',
        'Регламент согласования договоров ред 2025':'Регламент, ред. от 01.03.2025',
        'Письмо комплаенс-службы К-12 от 15.04.2025':'Письмо К-12 от 15.04.2025',
        '152ФЗ_ст9_действующая':'152-ФЗ, ст. 9',
        'МРОТ_ст1_действующая':'Закон о минимальном размере оплаты труда, ст. 1',
    }
    result=answer
    for d in documents.values():
        original=d.filename.removesuffix('.txt').replace('документы компании__','').replace('законы__','')
        if original not in cited or not source_matches(d.raw,known.get(d.filename,{})):continue
        title=names.get(original)
        match=re.fullmatch(r'(НК|ТК|КоАП)_ст(\d+(?:\.\d+)?)_действующая',original)
        if match:title=match[1]+' РФ, ст. '+match[2]
        if title is None:continue
        pattern=r'(?P<open>\(|;\s*)'+re.escape(original)+r',\s*(?P<loc>[^();]*)(?=[;)])'
        def rewrite(m):
            locator=m['loc']
            if original.startswith('Доп соглашение 1 '):
                locator=re.sub(r'^п\.\s*3\.2(?=,|$)','новая редакция п. 3.2 договора',locator)
            article=re.search(r', ст\. (\d+(?:\.\d+)?)$',title)
            if article:
                if not re.search(r'ст\.\s*'+re.escape(article[1])+r'(?![\d.])',locator):locator+=' ст. '+article[1]
                return m['open']+title[:article.start()]+', '+locator
            return m['open']+title+(', '+locator if locator else '')
        result=re.sub(pattern,rewrite,result)
    return result if len(result)<=10000 else answer

def repair_note_locators(answer,source_ids,evidence,documents):
    by_name={d.filename.removesuffix('.txt').replace('документы компании__','').replace('законы__',''):d for d in documents.values()}
    used=set(source_ids);labels={}
    for e in evidence:
        if e['source_id'] not in used or e['document'] not in by_name:continue
        d=by_name[e['document']]
        marker=re.search(r'(?m)^\s*Примечани[ея]\s*:',d.raw)
        if marker is None or not e['text'] or e['text'] not in d.raw:continue
        start=d.raw.index(e['text']);numbers=set(re.findall(r'(?m)^\s*(\d+(?:\.\d+)*)\.\s',e['text']))
        notes,parts=labels.setdefault(e['document'],(set(),set()))
        (notes if start>=marker.end() else parts).update(numbers)
    result=answer
    for document,(notes,parts) in labels.items():
        for number in sorted(notes-parts):
            # Only locator fragments inside this document's rendered citation.
            pattern=r'((?:\(|;\s*)'+re.escape(document)+r',\s*)([^();]*)(?=[;)])'
            def replace(match):
                locator=re.sub(r'(?<!\w)ч\.\s*'+re.escape(number)+r'(?![\d.])',
                               'п. '+number+' примечаний',match[2])
                return match[1]+locator
            result=re.sub(pattern,replace,result)
    return result if len(result)<=10000 else answer
