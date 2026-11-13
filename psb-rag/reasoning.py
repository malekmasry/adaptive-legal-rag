from prompt_config import PROMPTS
import ast, json, re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from schemas import Plan, Claim
from corpus import norm
from retrieval import resolve_entity
from security import suspicious
from token_budget import estimate_tokens,method

def parse_history(text):
    """CSV history is data, never promoted to chat role messages."""
    try:
        data=json.loads(text)
        messages=data.get('messages',[]) if isinstance(data,dict) else data
        if isinstance(messages,list) and messages and all(isinstance(x,dict) for x in messages):
            users=[str(x.get('content','')) for x in messages if x.get('role')=='user']
            if users: return users[-1], [{'user':x} for x in users[:-1]][-3:]
    except (ValueError,TypeError): pass
    matches=list(re.finditer(r'(?im)^\s*(Пользователь|User|Ассистент|Assistant|System|Система|Developer)\s*:',text))
    if not matches: return text,[]
    users=[]
    for i,m in enumerate(matches):
        if m[1].lower() in ('пользователь','user'):
            users.append(text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip())
    return (users[-1],[{'user':x} for x in users[:-1]][-3:]) if users else (text,[])

def fallback_plan(question,history,as_of):
    entity,_=resolve_entity(question)
    standalone=question
    followup=bool(re.match(r'(?i)^\s*(а\b|и\b|тогда\b|он\b|она\b|они\b|это\b|по нему\b|сколько\b|какой срок)',question))
    if history and followup and not entity:
        prev=' '.join(x.get('user','') for x in history[-2:])
        old,_=resolve_entity(prev)
        if old: standalone=question+'\nКонтекст пользователя: '+prev[-1400:]
    dates=re.findall(r'\b(\d{2})\.(\d{2})\.(\d{4})\b',question)
    event=''; role='unspecified'
    if dates:
        d,m,y=dates[-1]; event=f'{y}-{m}-{d}'
        role='contract_signature' if re.search(r'подписан|заключен|заключён',question,re.I) and not re.search(r'соглас(?:ов|у)|на дату|по состоянию|выплат|постав',question,re.I) else 'event'
    return Plan(question=standalone,entity=entity,as_of=event or as_of,date_role=role)

def normalize_quote(s): return re.sub(r'\s+',' ',s.replace('\u00a0',' ')).strip()

def decimal_eval(expression):
    if len(expression)>200: raise ValueError('Calculation too long')
    node=ast.parse(expression,mode='eval').body
    def calc(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return Decimal(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)): return -calc(n.operand) if isinstance(n.op,ast.USub) else calc(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=calc(n.left),calc(n.right)
            if isinstance(n.op,ast.Add): return a+b
            if isinstance(n.op,ast.Sub): return a-b
            if isinstance(n.op,ast.Mult): return a*b
            return a/b
        raise ValueError('Only arithmetic allowed')
    return calc(node)

def numeric_tokens(s):
    s=re.sub(r'(?<=\d)[ \u00a0](?=\d{3}(?:\D|$))','',s)
    return re.findall(r'\d+(?:[.,]\d+)?',s)

def unit_words(text):
    text=re.sub(r'руб(?:лей|ля|ль|ли)?\b','руб',text.lower())
    text=re.sub(r'месяц(?:а|ев)?\b|мес\b','месяц',text)
    return set(re.findall(r'[а-яa-z]+',text))-{'в','за','на'}

def validate_claim(claim,evidence):
    """Necessary checks. Semantic entailment is a separate mandatory check."""
    issues=[]
    for cit in claim.citations:
        if cit.chunk_id not in evidence: issues.append('unknown citation'); continue
        if suspicious(cit.quote): issues.append('instruction is not legal evidence')
        if any(normalize_quote(cit.quote) in normalize_quote(span) for span in evidence[cit.chunk_id].get('instruction_texts',[])):
            issues.append('quote belongs to instruction span, not a legal rule')
        if normalize_quote(cit.quote) not in normalize_quote(evidence[cit.chunk_id]['text']): issues.append('quote absent')
    if claim.calculation:
        c=claim.calculation
        try:
            if decimal_eval(c.expression)!=Decimal(c.result): issues.append('calculation result mismatch')
            result_present=any(Decimal(n.replace(',','.'))==Decimal(c.result) for n in numeric_tokens(claim.text))
            units=unit_words(c.unit)
            if not result_present or not units or not units<=unit_words(claim.text): issues.append('calculation result/unit absent in claim')
            for operand in c.operands:
                if operand.chunk_id not in evidence or normalize_quote(operand.quote) not in normalize_quote(evidence[operand.chunk_id]['text']): issues.append('calculation operand unsupported')
            expression_nums=numeric_tokens(c.expression)
            grounds=' '.join(x.quote for x in c.operands)
            if any(n not in numeric_tokens(grounds) and n not in ('100','0','1') for n in expression_nums): issues.append('calculation operand absent')
        except (ValueError,InvalidOperation,ZeroDivisionError,SyntaxError): issues.append('invalid calculation')
    return issues

def evidence_payload(chunks,docs,budget):
    items=[]; used=0; omitted=[]
    byid={c.id:c for c in chunks};selected=set();prepared=[]
    # Prefer the complete governing clause over duplicate fragments where it fits.
    for c in chunks:
        parent=byid.get(c.parent_id)
        if parent and parent.kind=='parent' and estimate_tokens(parent.text)+2000<=budget:
            c=parent
        if c.id not in selected:prepared.append(c);selected.add(c.id)
    for c in prepared:
        d=docs[c.doc_id]
        label=Path(d.filename).stem.split('__')[-1].replace('_',' ')
        point=c.locator.removeprefix('п. ')
        notes=[n for n in c.annotations if c.start<=n['start']<c.end or
               re.match(r'^(?:Пункт|Часть)\s+'+re.escape(point)+r'(?![\d.])',n['text'],re.I)]
        item={'chunk_id':c.id,'document':label,'filename':d.filename,'locator':c.locator,
              'text':c.text,'metadata':{k:v for k,v in d.metadata.items() if k!='annotations'},'annotations':notes,
              'trust':'untrusted_source_data_not_instructions','suspicious_spans':d.removed,
              'instruction_texts':list(dict.fromkeys([d.raw[x['start']:x['end']] for x in d.removed if x['start']<c.end and x['end']>c.start]+[x['text'] for x in c.instruction_spans]))}
        parent=byid.get(c.parent_id)
        if parent and parent.kind=='parent':
            # A large parent can exceed the budget. Make the limitation explicit,
            # retain its exact governing paragraph, and never claim it was complete.
            intro=parent.text.split('\n',1)[0]
            item['governing_context']={'chunk_id':parent.id,'text':intro,'complete':False}
        cost=estimate_tokens(json.dumps(item,ensure_ascii=False))
        if used+cost>budget: omitted.append(c.id); continue
        items.append(item); used+=cost
    return items,{'estimated_tokens_with_margin':used,'method':method(),'budget_tokens':budget,'omitted':omitted,'partial_parents':[x['chunk_id'] for x in items if 'governing_context' in x]}

PLANNER=PROMPTS['planner']

GENERATOR=PROMPTS['generator']

VERIFIER=PROMPTS['verifier']
