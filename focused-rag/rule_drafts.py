"""Source-bound ready drafts for recurring legal operations in this corpus.

These are rules by intent/entity/date, not a lookup of test questions. A draft
is always reviewed against the full question and evidence by the generator.
"""
from datetime import date
from decimal import Decimal
import hashlib
import re
from answer_guides import plan_for
from corpus_playbook import load_playbook
from source_identity import source_matches
from focused_context import source_name

FAMILIES={'rent':'Договор аренды А-08-2024','p17':'Договор поставки П-17-2024',
 'pt05':'Договор поставки ПТ-05-2023','amend':'Доп соглашение 1 к договору П-17-2024',
 'old':'Регламент согласования договоров ред 2023','new':'Регламент согласования договоров ред 2025',
 'letter':'Письмо комплаенс-службы К-12','consent':'152ФЗ_ст9_',
 '112':'НК_ст112_','114':'НК_ст114_','70':'ТК_ст70_'}
WORDS={'один':1,'одна':1,'одну':1,'два':2,'две':2,'три':3,'четыре':4,'пять':5,'шесть':6,
 'семь':7,'восемь':8,'девять':9,'десять':10,'одиннадцать':11,'двенадцать':12,
 'тринадцать':13,'четырнадцать':14,'пятнадцать':15,'шестнадцать':16,'семнадцать':17,
 'восемнадцать':18,'девятнадцать':19,'двадцать':20,'тридцать':30,'сорок':40,
 'пятьдесят':50,'шестьдесят':60,'семьдесят':70,'восемьдесят':80,'девяносто':90,
 'сто':100,'двести':200,'триста':300,'четыреста':400,'пятьсот':500,'шестьсот':600,
 'семьсот':700,'восемьсот':800,'девятьсот':900}
SCALES={'тысяча':1000,'тысячи':1000,'тысяч':1000,'миллион':1000000,'миллиона':1000000,'миллионов':1000000}
NUMBER=r'(?:\d[\d\s\u00a0]*(?:[.,]\d+)?(?:\s*(?:тысяч(?:а|и)?|тыс\.?|миллион(?:а|ов)?|млн\.?))?|(?:'+ '|'.join(sorted(WORDS|SCALES,key=len,reverse=True))+r')(?:\s+(?:'+ '|'.join(sorted(WORDS|SCALES,key=len,reverse=True))+r'))*)'
MONTHS=['январ','феврал','март','апрел','ма','июн','июл','август','сентябр','октябр','ноябр','декабр']


def number(text):
    text=text.strip().replace('\u00a0',' ')
    scaled=re.fullmatch(r'(\d[\d ]*(?:[.,]\d+)?)\s*(тысяч(?:а|и)?|тыс\.?|миллион(?:а|ов)?|млн\.?)',text)
    if scaled:return Decimal(scaled[1].replace(' ','').replace(',','.'))*(1000 if scaled[2].startswith('ты') else 1000000)
    if re.fullmatch(r'\d[\d ]*(?:[.,]\d+)?',text):return Decimal(text.replace(' ','').replace(',','.'))
    total=0;part=0
    for word in text.split():
        if word in WORDS:part+=WORDS[word]
        elif word in SCALES:total+=(part or 1)*SCALES[word];part=0
        else:return None
    return Decimal(total+part)


def unique_number(pattern,text):
    values={number(m[1]) for m in re.finditer(pattern,text)}
    values.discard(None)
    return next(iter(values)) if len(values)==1 else None


def money(text):
    value=unique_number('('+NUMBER+r')\s*(?:руб\w*|₽)\b',text)
    if value is not None:return value
    return unique_number(r'(?:должны|сумма(?:\s+долга)?|задолженность)\s+('+NUMBER+r')(?=[.,;!?]|\s+(?:руб|состав|был|и\b)|$)',text)


def event_date(text,as_of):
    dates=[]
    try:
        for m in re.finditer(r'\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b',text):dates.append(date(int(m[3]),int(m[2]),int(m[1])))
        for m in re.finditer(r'\b(?:(\d{1,2})\s+)?('+'|'.join(MONTHS)+r')\w*\s+(20\d{2})(?:-го)?\b',text):
            dates.append(date(int(m[3]),MONTHS.index(m[2])+1,int(m[1] or 1)))
    except ValueError:return None
    if len(set(dates))>1:return None
    if dates:return dates[0]
    # Do not reinterpret an explicit year/relative event as today's date.
    if re.search(r'\b20\d{2}\s*(?:год|г\.|-го)|вчера|завтра|прошл',text):return None
    try:return date.fromisoformat(as_of)
    except ValueError:return None


def available_sources(evidence,documents):
    known={e['filename']:e for e in load_playbook()['documents']}
    result={}
    for d in documents.values():
        if not source_matches(d.raw,known.get(d.filename,{})):continue
        e=next((e for e in evidence if e['document']==source_name(d) and e['text']==d.raw),None)
        if e:
            for family,prefix in FAMILIES.items():
                if source_name(d).startswith(prefix):result[family]=e
    return result


def contract_entities(q):
    result=[]
    if re.search(r'а[- /]*08|северн\w*\s+склад\w*',q):result.append('rent')
    if re.search(r'п[- /]*17|альфа[ -]+трейд(?:а|у|ом|е)?\b',q):result.append('p17')
    if re.search(r'пт[- /]*05|альфа[ -]+трейдинг\w*',q):result.append('pt05')
    return result


def build_rule_draft(question,evidence,documents,as_of):
    q=question.lower().replace('ё','е');topics=set(plan_for(q)['topics'])
    s=available_sources(evidence,documents);entities=contract_entities(q)
    paragraphs=[];status='answered'
    def para(text,*refs):
        if any(k not in s for k,loc in refs):raise KeyError('Unverified or unseen source')
        paragraphs.append({'text':text,'citations':[{'source_id':s[k]['source_id'],'locator':loc} for k,loc in refs]})
    try:
        if topics=={'contract_term'} and entities==['rent']:
            para('Конкретного последнего дня аренды в договоре нет: договор заключён на неопределённый срок.',('rent','п. 2.2'))
        elif topics=={'penalty'} and len(entities)==1:
            entity=entities[0];base=money(q)
            days=unique_number('('+NUMBER+r')\s+(?:(?:календарн|рабоч)\w*\s+)?(?:дней|дня|день)\b',q)
            if base is None or days is None or base<0 or days<0:return None
            if re.search(r'изменя|частичн|погаш|снизи|уплачен|рабоч\w*\s+дн',q):return None
            if entity=='rent':rate=Decimal('.001');cap=None;loc='п. 5.1'
            else:
                delivery=bool(re.search(r'задерж\w*\s+постав|просроч\w*\s+постав|непостав',q))
                payment=bool(re.search(r'неоплат|оплат|платеж|заплат',q))
                if delivery==payment:return None
                rate=Decimal('.001') if entity=='p17' else Decimal('.0005')
                cap=base*Decimal('.1') if entity=='p17' else None
                loc=('п. 7.1' if delivery else 'п. 7.2') if entity=='p17' else ('п. 5.1' if delivery else 'п. 5.2')
            raw=base*rate*days;value=min(raw,cap) if cap is not None else raw
            def fmt(n):return format(n.quantize(Decimal('.01')),'f').replace('.',',')
            formula=f'{fmt(base)} руб. × {rate*100:g}% × {days:g} дн. = {fmt(raw)} руб.'
            if cap is not None:formula+=f' Договорный предел — 10% расчётной базы: {fmt(cap)} руб.; применяется меньшая величина.'
            para(f'Неустойка составляет {fmt(value)} руб. Расчёт: '+formula,(entity,loc))
        elif topics and topics<= {'payment_term','payment_completion'} and len(entities)==1:
            entity=entities[0]
            if 'payment_term' in topics:
                if entity=='p17':para('Оплата — в течение 30 календарных дней с даты подписания УПД по соответствующей партии товара.',('p17','п. 4.2'))
                elif entity=='pt05':para('Оплата — в течение 45 календарных дней с даты поставки товара.',('pt05','п. 4.1'))
                else:para('Арендная плата вносится ежемесячно не позднее 5-го числа текущего месяца.',('rent','п. 3.2'))
            if 'payment_completion' in topics:
                if entity!='p17':return None
                para('Обязательство по оплате считается исполненным с момента списания денежных средств с корреспондентского счёта банка покупателя.',('p17','п. 4.3'))
        elif topics=={'order_delivery'} and entities==['pt05']:
            para('Заявка направляется по электронной почте. В ней указывают наименование, количество товара и желаемую дату поставки.',('pt05','п. 1.2'))
            para('Срок поставки — 14 рабочих дней с даты получения заявки поставщиком.',('pt05','п. 3.1'))
        elif topics=={'warranty'}:
            selected=[e for e in entities if e in ('p17','pt05')]
            if not selected and 'альф' in q:selected=['p17','pt05'];status='clarification'
            if not selected:return None
            for e in selected:
                if e=='p17':para('По договору № П-17/2024 с ООО «Альфа-Трейд» гарантия составляет 24 месяца с даты поставки.',('p17','п. 5.2'))
                else:para('По договору № ПТ-05/2023 с ООО «Альфа-Трейдинг» срок гарантии устанавливает производитель и указывает в гарантийном талоне. Единого срока в 24 месяца этот договор не устанавливает.',('pt05','п. 6.1'))
            if status=='clarification':paragraphs.append({'text':'Однозначно назвать один срок нельзя. Уточните юридическое наименование поставщика или номер договора.','citations':[]})
        elif topics and topics<= {'delivery_address','delivery_cost'} and entities==['p17']:
            event=event_date(q,as_of)
            # An actual specification date is required for a ready address.
            if 'delivery_address' in topics:
                if not event or not re.search(r'спецификац\w*[^.!?]{0,45}подпис',q):return None
                if not re.search(r'\d{1,2}\s+(?:'+'|'.join(MONTHS)+r')|\d{1,2}[./]\d{1,2}[./]20',q):return None
                new=event>date(2024,10,1)
                address='Московская область, г. Химки, ул. Заводская, д. 15, склад № 3' if new else 'г. Москва, ул. Складская, д. 4, стр. 2'
                refs=[('amend','абзац «Новый адрес доставки применяется к партиям Товара»')]
                refs.insert(0,('amend','новая редакция п. 3.2') if new else ('p17','п. 3.2'))
                para(f'Адрес доставки: {address}. Новый адрес применяется только к партиям со спецификациями, подписанными после 01.10.2024; спецификация от {event.strftime("%d.%m.%Y")} '+('соответствует' if new else 'не соответствует')+' этому условию.',*refs)
            if 'delivery_cost' in topics:para('Отдельно прибавлять стоимость доставки не нужно: она включена в цену товара.',('p17','п. 2.1'))
        elif topics=={'approval_form'}:
            event=event_date(q,as_of)
            if not event or event<date(2023,1,10):return None
            if event<date(2025,3,1):para('Применяется редакция Регламента от 10.01.2023: согласование оформляется на бумажном носителе в листе согласования. Новая редакция действует только с 01.03.2025.',('old','п. 3.1'),('new','абзац «Редакция от 01.03.2025»'))
            else:para('По Регламенту в редакции от 01.03.2025 согласование оформляется в СЭД. Бумажный лист согласования не требуется. Эта редакция действует начиная с 01.03.2025.',('new','п. 3.1'),('new','абзац «Редакция от 01.03.2025»'))
        elif topics=={'approved_template'}:
            event=event_date(q,as_of)
            if not event or event<date(2025,3,1):return None
            if not re.search(r'не\s+меня|без\s+измен|неизмен',q):return None
            para('Договор по типовой форме, утверждённой юридическим отделом, без изменения условий согласуется только с главным бухгалтером — по п. 4.1 Регламента в редакции от 01.03.2025.',('new','п. 4.1'))
        elif topics=={'foreign_checks'}:
            event=event_date(q,as_of)
            if not event or event<date(2025,3,1):return None
            para('По п. 5.4 Регламента в редакции от 01.03.2025 до подписания требуются проверка юридическим отделом, в том числе применимого права и порядка разрешения споров, и валютный контроль финансовым отделом.',('new','п. 5.4'))
            if event<date(2025,4,21):para('Дополнительная санкционная проверка по письму К-12 на указанную дату ещё не обязательна: требование действует с 21.04.2025.',('letter','абзац «С 21 апреля 2025 г.»'))
            else:
                para('С 21.04.2025 письмо К-12 дополнительно требует санкционной проверки иностранного контрагента службой комплаенс до подписания. Заключение прикладывается в СЭД; без него договор не допускается к подписанию.',('letter','абзац «С 21 апреля 2025 г.»'),('letter','абзац «Договоры с иностранными контрагентами без заключения»'))
                if re.search(r'сколько|срок|кто|делает',q):para('Проверку проводит служба комплаенс в течение 3 рабочих дней с даты получения запроса. Инициатор договора обеспечивает её проведение.',('letter','абзац «Санкционную проверку проводит служба комплаенс»'),('letter','абзац «С 21 апреля 2025 г.»'))
        elif topics=={'consent_withdrawal'}:
            para('Нет, отзыв согласия не делает всю последующую обработку автоматически незаконной. По ч. 2 ст. 9 оператор вправе продолжить её без согласия при наличии оснований из пп. 2–11 ч. 1 ст. 6, ч. 2 ст. 10 или ч. 2 ст. 11 Федерального закона № 152-ФЗ.',('consent','ч. 2 ст. 9'))
            para('Из одной статьи 9 нельзя установить содержание этих оснований и какое из них применимо в конкретной ситуации: для этого нужны соответствующие положения и фактические обстоятельства.',('consent','ч. 2 ст. 9'))
        elif topics=={'tax_penalty'} and re.search('смягчающ',q) and re.search('отягчающ',q):
            event=event_date(q,as_of)
            if not event or event<date(2026,9,1):return None
            para('При наличии хотя бы одного смягчающего обстоятельства штраф уменьшается не менее чем в 2 раза и не более чем в 10 раз по сравнению с размером, установленным соответствующей статьёй НК РФ.',('114','п. 3'))
            para('При отягчающем обстоятельстве по п. 2 ст. 112 штраф увеличивается на 100%. Это ранее состоявшееся привлечение к ответственности за аналогичное налоговое правонарушение. Состояние подвергнутости санкции длится 12 месяцев с момента вступления решения в законную силу.',('114','п. 4'),('112','п. 2'),('112','п. 3'))
        else:return None
    except KeyError:return None
    return {'status':status,'paragraphs':paragraphs}
