"""Source-bound decision paths, contrasting examples, and targeted review traps."""
import json,re
from functools import lru_cache
from pathlib import Path
from source_identity import source_matches,live_span
from focused_context import source_name
from answer_guides import plan_for
from retrieval import tokens
from token_budget import estimate_tokens
from rule_drafts import contract_entities

GUIDANCE='''scenario_support — проверенные по текущим источникам сценарии с переменными, учебными примерами и ловушками. Для выбранного сценария сопоставь inputs_to_bind с фактами текущего вопроса, затем проверь нужную ветку branches по evidence. Неизвестное условие не считается выполненным; явно заданное выполненным не оспаривай. Ветви описывают разные случаи, а не дополнительные вопросы пользователя. Используй только относящиеся к запрошенному выводу.
contrast_examples — гипотетические примеры применения того же правила, НЕ обстоятельства пользователя и НЕ доказательства. Их числа, даты и результаты нельзя переносить в ответ. Если изменилось условие примера, заново выбери ветку и пересчитай результат по фактам вопроса. Для составного вопроса свяжи относящиеся сценарии, сохранив каждый запрошенный вывод. Основание факта и ссылки всегда берутся из evidence; suggested_sources — проверенные места источника, а не готовое подтверждение любого вывода.
review_traps — конкретные возможные ошибки. Проверь их только применительно к вопросу и черновику; не пересказывай пользователю этот список. Если вычисление уже дано в decision_checks, проверь его предпосылки и используй относящийся результат. Дай один прямой вывод на каждый смысловой подвопрос; не повторяй ту же сумму/предел отдельным абзацем без нового содержания. Не выводи сценарии, названия полей и учебные примеры в ответ.'''


@lru_cache(maxsize=1)
def load_bank():
    return json.loads(Path(__file__).with_name('scenario_bank.json').read_text(encoding='utf8'))['scenarios']


def bind_sources(row,documents,evidence):
    by_name={d.filename:d for d in documents.values()};refs=[]
    for source in row['sources']:
        doc=by_name.get(source['filename'])
        if doc is None or not source_matches(doc.raw,source):return None
        visible=[e for e in evidence if e['document']==source_name(doc)]
        for anchor in source['anchors']:
            span=live_span(doc.raw,anchor['text'])
            item=next((e for e in visible if span and span[2] in e['text']),None)
            if item is None:return None
            value={'source_id':item['source_id'],'locator':anchor['locator']}
            if value not in refs:refs.append(value)
    return refs or None


def support(question,evidence,documents,budget,bank=None,focuses=None):
    if budget<=0:return [],0,{'selected':[],'eligible':0}
    q=question.lower().replace('ё','е');topics=set(plan_for(q)['topics']);query=set(tokens(q))
    focus=' '.join(focuses or []).lower().replace('ё','е')
    focus_topics=set(plan_for(focus)['topics']) if focus else set()
    entities=set(contract_entities(focus))
    if re.search(r'аренд',focus):entities.add('rent')
    ranked=[]
    for row in load_bank() if bank is None else bank:
        if not re.search(row['when'],q):continue
        if row.get('focus_when') and not re.search(row['focus_when'],focus or q):continue
        overlap=topics.intersection(row['topics'])
        # The locally authored topic matcher must agree; a number appearing in
        # an unrelated question cannot activate a legal decision by itself.
        if not overlap:continue
        if entities and row.get('entity') and row['entity'] not in entities:continue
        if row['id']=='foreign_checks' and re.search(r'нет иностранного контрагента|не иностранн',q):continue
        refs=bind_sources(row,documents,evidence)
        if refs is None:continue
        words=' '.join(row['branches'])
        score=30*len(focus_topics.intersection(row['topics']))+10*len(overlap)+len(query.intersection(tokens(words)))
        ranked.append((score,row,refs,overlap))
    ranked.sort(key=lambda x:(-x[0],x[1]['id']))
    order=ranked[:1]
    for topic in sorted(focus_topics or topics):
        candidate=next((x for x in ranked if topic in x[3]),None)
        if candidate and candidate[1]['id'] not in [x[1]['id'] for x in order]:order.append(candidate)
    order.extend(x for x in ranked if x[1]['id'] not in {r[1]['id'] for r in order})
    selected=[]
    # Count framing as well as JSON; never take budget from source evidence.
    def cost(rows):return estimate_tokens(GUIDANCE)+estimate_tokens(json.dumps(rows,ensure_ascii=False))
    for _,row,refs,_ in order:
        card={'scenario':row['id'],'inputs_to_bind':row['slots'],'branches':row['branches'],
              'contrast_examples':row['contrast_examples'],'review_traps':row['review_traps'],
              'suggested_sources':refs}
        if cost(selected+[card])>budget:continue
        selected.append(card)
        if len(selected)>=3:break
    return selected,cost(selected) if selected else 0,{'selected':[r['scenario'] for r in selected],'eligible':len(ranked)}
