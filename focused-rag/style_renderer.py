"""Deterministic legal-answer presentation, preserving claims and citations.

Style policy is by legal question type, never question ID or expected answer.
All modes retain every answer paragraph, source locator and temporal scope.
"""
from citation_labels import display_citations

STYLE_POLICY={}
STYLES=('plain_scope','compact_scope','compact_today','compact_meta','inline_scope','inline_today',
        'bullets_scope','numbered_scope','grounds_scope','emphasis_scope','table_scope','table_today')

def topic_group(topics):
    selected=set(topics)
    if selected=={'warranty'}:return 'warranty'
    if selected=={'probation'}:return 'probation'
    if selected=={'foreign_checks'}:return 'foreign_checks'
    if selected=={'consent_withdrawal'}:return 'consent_withdrawal'
    for topic in ('contract_term','approval_threshold','approval_form','approved_template','delivery_address','tax_penalty'):
        if selected=={topic}:return topic
    if selected and selected<={'payment_term','payment_completion'}:return 'payment'
    return None

def apply_policy(answer,topics,source_ids,evidence,documents):
    style=STYLE_POLICY.get(topic_group(topics),'plain_scope')
    return render_style(answer,source_ids,evidence,documents,style)

def render_style(answer,source_ids,evidence,documents,style):
    if style not in STYLES:raise ValueError('Unknown answer style')
    if style=='plain_scope':return answer
    layout,scope=style.split('_')
    original=answer;note='По состоянию на текущую дату.'
    has_scope=answer.endswith('\n\n'+note)
    if has_scope:answer=answer[:-len('\n\n'+note)]
    names={e['document'] for e in evidence};rows=[]
    for p in answer.split('\n\n'):
        indices=[p.find(' ('+n+', ') for n in names];indices=[i for i in indices if i>=0]
        if indices and p.endswith(')'):
            i=min(indices);body=p[:i];citation=p[i+1:]
            short=display_citations(citation,source_ids,evidence,documents)
            rows.append((body,short[1:-1]))
        else:rows.append((p,''))
    inline=[body+(' ('+ref+')' if ref else '') for body,ref in rows]
    if layout=='compact':result='\n\n'.join(inline)
    elif layout=='inline':result=' '.join(inline)
    elif layout=='bullets':result='\n\n'.join('— '+p for p in inline)
    elif layout=='numbered':result='\n\n'.join(str(i+1)+'. '+p for i,p in enumerate(inline))
    elif layout=='grounds':result='\n\n'.join(body+('\nОснование: '+ref+'.' if ref else '') for body,ref in rows)
    elif layout=='emphasis':result='\n\n'.join('**'+body+'**'+(' ('+ref+')' if ref else '') for body,ref in rows)
    elif layout=='table':
        def cell(s):return s.replace('|','\\|').replace('\n',' ')
        result='| Ответ | Основание |\n|---|---|\n'+'\n'.join('| '+cell(body)+' | '+cell(ref)+' |' for body,ref in rows)
    else:raise ValueError('Unknown layout')
    if has_scope:
        footer={'scope':note,'today':'Вывод дан на текущую дату.','meta':'Актуальность ответа: на текущую дату.'}[scope]
        result+='\n\n'+footer
    return result if len(result)<=10000 else original
