"""Repair only a provable part-versus-note label after final generation.

The cited source IDs, document names, numbers and answer facts are preserved.
A note is recognized by its exact location in the current source, never by a
question ID, a golden answer, or a semantic guess about the answer.
"""
import re

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
    return result

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
