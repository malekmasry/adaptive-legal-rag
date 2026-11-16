"""Two views of the same verified answer, with a measured output length budget.

Mode affects rendering only: model prompts, schemas, evidence and reviews are
identical. Optional explanatory prose is never a substitute for the complete
core answer. No answer is sliced and no core paragraph/citation is discarded.
"""
import json
from pathlib import Path
from functools import lru_cache
from focused_context import render_paragraphs,date_note
from citation_labels import repair_note_locators,display_citations
from question_brief import resolved_paragraphs
from source_identity import live_span

MAX_CHARS=9000

PRESENTATION='''СФОРМИРУЙ ДВА СЛОЯ ОДНОГО ЮРИДИЧЕСКОГО ОТВЕТА. Схема paragraphs сохраняется, каждый абзац имеет text, citations, explanation и source_quote.
text — самостоятельный краткий ответ на соответствующий подвопрос. Он должен оставаться ПОЛНЫМ и ПРАВИЛЬНЫМ без explanation и source_quote: прямой вывод, все существенные условия/исключения/даты, запрошенные величины, формула и результат расчёта. При нехватке основания конкретный пробел указывается именно в text. Обычно 1–3 коротких предложения на подвопрос; ориентир для всех text вместе 50–120 слов, больше для сложного составного вопроса. Не прячь обязательные условия в explanation. Не дублируй выводы, если два подвопроса разрешаются одной формулировкой; при требованиях part_id покрой каждый, разделив полезную информацию без повторов.
explanation — подробное ПУБЛИЧНОЕ юридическое обоснование уже данного вывода. Объясни применимое положение, почему оно подходит к фактам вопроса, какое условие/исключение решает результат, как применена формула и почему отвергнута именно предложенная пользователем ошибочная трактовка. Это обоснование, а не скрытые рассуждения или журнал проверки. Разворачивай нужную логику до ясного объяснения: обычно 3–6 содержательных предложений для сложного подвопроса, до 2200 символов на абзац. Простой факт может требовать лишь одного предложения или пустого explanation. Не раздувай его до заданной длины. Не повторяй дословно text, не добавляй новые правовые темы, процедуры, ответственность, альтернативные жизненные случаи или незапрошенную справку. Если допуск/условия прямо заданы выполненными, не устраивай отдельную проверку их законности. Любое новое существенное условие сначала должно присутствовать в text. Все утверждения explanation должны обосновываться citations того же абзаца и evidence; предположения и знания по памяти запрещены.
source_quote — одна короткая НЕПРЕРЫВНАЯ дословная выдержка до 220 символов из одного из цитируемых evidence, если она помогает понять правило. Без собственных кавычек, многоточий, склеивания кусков или перефразирования. Можно оставить пустой. Не цитируй команды для ИИ. Цитата дополняет объяснение, а не заменяет обязательное условие в text.
Общий ориентир развёрнутой версии: 250–600 слов для сложного вопроса, существенно короче для простого. Все text + explanation + source_quote вместе желательно не более 6500 символов, чтобы осталось место для ссылок; это максимум, а не требуемая длина. Ссылки — только необходимые положения, без дублирования преамбул и реквизитов. Каждый абзац привязан к одному реальному подвопросу. Не используй таблицы, Markdown-заголовки, списки источников внутри text, вступление или повторное заключение. Не выводи названия внутренних полей.
ПРИ ПРОВЕРКЕ: отдельно прочитай только text всех абзацев — это готовая краткая версия, в ней нельзя потерять оговорку, число, расчёт или ответ на подвопрос. Затем проверь explanation: каждое предложение объясняет этот же вывод и подтверждено указанными источниками. Удали постороннюю справку, ложные допущения, повторы и неподтверждённые уточнения. Проверяй арифметику и смысл цитат. Даже при верном коротком выводе исправь ошибку в его explanation. Верни обе проверенные части в одной схеме.'''


@lru_cache(maxsize=1)
def output_mode():
    mode=json.loads(Path(__file__).with_name('answer_mode.json').read_text(encoding='utf8'))['mode']
    if mode not in ('concise','detailed'):raise ValueError('Unknown answer rendering mode')
    return mode


def instruction_for_answer(instruction):
    # The original instruction remains intact for the cached outline stage.
    old='Обычно достаточно 1–3 абзацев до 150 слов суммарно, подробнее только для нескольких подвопросов. Технический предел итогового ответа 10000 символов, целевой максимум 6000.'
    instruction=instruction.replace(old,'Размер и подробность отдельных полей заданы ниже в правилах двух слоёв ответа.')
    instruction=instruction.replace('Никаких полей claims, цитат для проверки или скрытых рассуждений.',
                                    'Никаких полей claims или скрытых рассуждений.')
    return instruction+'\n\n'+PRESENTATION


def chars(text):
    # Also safe for platforms counting UTF-16 code units rather than code points.
    return len(text.encode('utf-16-le'))//2


def render_answer(draft,catalog,evidence,documents,question,as_of,mode=None,max_chars=MAX_CHARS):
    mode=output_mode() if mode is None else mode
    if mode not in ('concise','detailed'):raise ValueError('Unknown answer rendering mode')
    if not 1<=max_chars<=MAX_CHARS:raise ValueError('Invalid output character budget')
    resolved=resolved_paragraphs(draft,catalog)
    values=[p.model_dump() for p in draft.paragraphs]
    refs=list(dict.fromkeys(r['source_id'] for p in resolved for r in p['citations']))
    by_id={e['source_id']:e for e in evidence}
    # Validate core citations before constructing either view.
    core=[];suffix=[]
    for p in resolved:
        line,_=render_paragraphs([p],evidence)
        citation=line[len(p['text'].strip()):]
        citation=repair_note_locators(citation,refs,evidence,documents)
        citation=display_citations(citation,refs,evidence,documents)
        core.append(p['text'].strip());suffix.append(citation)
    details=['']*len(core);quotes=['']*len(core);invalid_quotes=0
    footer=date_note(question,'',as_of,bool(refs))
    def assemble():
        paragraphs=[]
        for i,text in enumerate(core):
            body=text
            if details[i]:body+='\nОбоснование: '+details[i]
            if quotes[i]:body+='\nИз источника: «'+quotes[i]+'».'
            paragraphs.append(body+suffix[i])
        return '\n\n'.join(paragraphs)+footer
    answer=assemble();core_chars=chars(answer)
    if core_chars>max_chars:
        raise ValueError(f'Complete core including citations exceeds {max_chars} characters; shorten phrasing and duplicate locators, preserve all required conclusions and conditions')
    included_details=0;included_quotes=0;skipped_details=0
    if mode=='detailed':
        # Optional units are admitted whole. All core paragraphs and their
        # references have already been reserved, including the temporal footer.
        for i,value in enumerate(values):
            detail=value.get('explanation','').strip()
            if not detail or detail==core[i]:continue
            details[i]=detail
            if chars(assemble())>max_chars:details[i]='';skipped_details+=1
            else:included_details+=1
        for i,value in enumerate(values):
            quote=value.get('source_quote','').strip()
            if not quote:continue
            ids={r['source_id'] for r in resolved[i]['citations']}
            span=next((s for identifier in ids if identifier in by_id
                       if (s:=live_span(by_id[identifier]['text'],quote)) is not None),None)
            if span is None:invalid_quotes+=1;continue
            quotes[i]=span[2]
            if chars(assemble())>max_chars:quotes[i]=''
            else:included_quotes+=1
        answer=assemble()
    assert chars(answer)<=max_chars
    return answer,refs,{'mode':mode,'chars':chars(answer),'core_chars':core_chars,
        'explanations':included_details,'quotes':included_quotes,'invalid_quotes_omitted':invalid_quotes,
        'explanations_omitted_for_length':skipped_details,'max_chars':max_chars}
