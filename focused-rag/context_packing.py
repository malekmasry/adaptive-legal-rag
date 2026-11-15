"""Budget fallback: complete retrieved scopes first, full documents second.

Never edit a source or sever a legal paragraph to fit. A retrieved child and
its available semantic parent form one allocation unit. Used only when the
original full-document pack would drop text.
"""
import json
from token_budget import estimate_tokens

def merge(spans):
    result=[]
    for start,end in sorted(set(spans)):
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result

def pack_scopes(candidates,chunks,budget):
    order=list(dict.fromkeys(d.id for d,scope,text in candidates))
    docs={d.id:d for d,scope,text in candidates}
    by_id={getattr(c,'id',None):c for c in chunks if getattr(c,'id',None)}
    groups={i:[] for i in order};upgrades={};invalid=0
    def span(c,d):
        start,end=getattr(c,'start',None),getattr(c,'end',None)
        if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(d.raw):return None
        if hasattr(c,'text') and c.text!=d.raw[start:end]:return None
        return start,end
    for identifier in order:
        d=docs[identifier]
        for c in chunks:
            if c.doc_id!=identifier or getattr(c,'kind','passage')=='parent':continue
            child=span(c,d)
            if child is None:continue
            atoms=[child];parent=by_id.get(getattr(c,'parent_id',''))
            if parent is not None:
                parent_span=span(parent,d)
                if parent_span is None:invalid+=1;continue
                atoms.append(parent_span)
            group=tuple(merge(atoms))
            if group not in groups[identifier]:groups[identifier].append(group)
        # A linked projection without retrieved chunks remains an intact unit.
        if not groups[identifier]:
            for cd,scope,text in candidates:
                if cd.id!=identifier:continue
                start=d.raw.find(text)
                if start<0:invalid+=1;continue
                group=((start,start+len(text)),)
                if group not in groups[identifier]:groups[identifier].append(group)
        if any(cd.id==identifier and text==d.raw for cd,scope,text in candidates):upgrades[identifier]=[(0,len(d.raw))]
    def serialize(selected):
        evidence=[];cost=0
        for identifier in order:
            d=docs[identifier]
            name=d.filename.removesuffix('.txt').replace('документы компании__','').replace('законы__','')
            for start,end in merge(selected.get(identifier,[])):
                scope='полный документ' if start==0 and end==len(d.raw) else f'фрагмент, символы {start}:{end}'
                item={'source_id':'D'+str(len(evidence)+1),'document':name,'title':d.title,'scope':scope,'text':d.raw[start:end]}
                cost+=estimate_tokens(json.dumps(item,ensure_ascii=False));evidence.append(item)
        return evidence,cost
    selected={};accepted=0;rejected=[]
    # Round-robin prevents optional full text of the first source from replacing
    # a retrieved provision from the next source. Parent conditions stay intact.
    for index in range(max((len(v) for v in groups.values()),default=0)):
        for identifier in order:
            if index>=len(groups[identifier]):continue
            group=groups[identifier][index]
            trial={**selected,identifier:merge(selected.get(identifier,[])+list(group))}
            if serialize(trial)[1]<=budget:selected=trial;accepted+=1
            else:rejected.append({'document':docs[identifier].filename,'spans':[list(s) for s in group]})
    # Upgrade only using remaining space, never evicting already placed sources.
    for identifier,whole in upgrades.items():
        trial={**selected,identifier:whole}
        if serialize(trial)[1]<=budget:selected=trial
    evidence,cost=serialize(selected)
    full={i for i,spans in selected.items() if merge(spans)==[(0,len(docs[i].raw))]}
    return evidence,{'estimated_tokens':cost,'budget':budget,'omitted_documents':[docs[i].filename for i in order if not selected.get(i)],
        'partial_documents':[docs[i].filename for i in order if selected.get(i) and i not in full],'full_documents':len(full),
        'packing_strategy':'complete_retrieved_scopes_then_full_documents','packing_units_accepted':accepted,
        'packing_units_unallocated':rejected,'invalid_source_spans':invalid}
