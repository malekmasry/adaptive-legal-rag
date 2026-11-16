"""Independent retrieval over source-bound decisions and paraphrased questions.

All source excerpts remain evidence. Only the existing reviewed playbook may
produce system hints. An unknown/changed document keeps ordinary hybrid RAG.
"""
import hashlib,json,re
from pathlib import Path
from rank_bm25 import BM25Okapi
from answer_guides import plan_for
from intent_rules import QUESTIONS
from retrieval import tokens,exact_entities
from corpus_playbook import load_playbook
from token_budget import estimate_tokens
from source_identity import source_matches,live_span,live_scope,fingerprints

def source_name(d):return d.filename.removesuffix('.txt').replace('документы компании__','').replace('законы__','')

class DecisionIndex:
    def __init__(self,documents,catalog=None,playbook=None):
        catalog=json.loads(Path(__file__).with_name('decision_catalog.json').read_text(encoding='utf8')) if catalog is None else catalog
        book=load_playbook() if playbook is None else playbook
        self.docs=documents;self.rows=[]
        by_name={d.filename:d for d in documents.values()}
        self.valid={d['id'] for d in catalog['documents'] if d['id'] in documents
            and documents[d['id']].filename==d['filename']
            and source_matches(documents[d['id']].raw,d)}
        self.exact={d['id'] for d in catalog['documents'] if d['id'] in self.valid
            and fingerprints(documents[d['id']].raw)[0]==d['sha256']}
        for entry in book['documents']:
            d=by_name.get(entry['filename'])
            if d is None or d.id not in self.valid or not source_matches(d.raw,entry):continue
            for i,n in enumerate(entry['notes']):
                spans=[live_span(d.raw,a) for a in n['anchors']]
                if not all(spans):continue
                self.rows.append({'id':d.id+':note:'+str(i),'doc_id':d.id,'topics':n['topics'],
                    'when':n.get('when'),'spans':[s[2] for s in spans],'locator':'основания проверенной памятки',
                    'search':n['hint'],'kind':'reviewed_note'})
        seen=set()
        for unit in catalog['units']:
            if unit['doc_id'] not in self.valid:continue
            d=documents[unit['doc_id']];scope=live_scope(d.raw,unit,d.id in self.exact)
            if scope is None:continue
            a,b,_=scope
            # Equal full scopes share one index row, including all child wording.
            if (d.id,a,b) in seen:continue
            seen.add((d.id,a,b))
            self.rows.append({'id':unit['id'],'doc_id':d.id,'topics':unit['topics'],'when':None,
                'spans':[d.raw[a:b]],'locator':unit['locator'],'search':d.raw[a:b], 'kind':'source_scope'})
        texts=[]
        for row in self.rows:
            d=documents[row['doc_id']]
            aliases=' '.join(q for t in row['topics'] for q in QUESTIONS.get(t,[]))
            texts.append(tokens(source_name(d)+' '+row['search']+' '+aliases))
        self.bm=BM25Okapi(texts) if texts else None

    def select(self,question,topics=None,budget=6000):
        if self.bm is None or budget<=0:return [],{'selected':[],'tokens':0}
        q=question.lower().replace('ё','е');topics=list(topics or plan_for(q)['topics'])
        articles=re.findall(r'(?:стать[еяию]|ст\.)\s*(\d+(?:[.,]\d+)?)',q)
        if articles and not topics and not any('_ст'+a.replace(',','.')+'_' in d.filename for a in articles for d in self.docs.values()):
            return [],{'selected':[],'tokens':0,'reason':'explicit_article_absent_from_catalog'}
        entities=set(exact_entities(question));scores=self.bm.get_scores(tokens(question))
        # Corpus-specific name disambiguation complements literal YAML aliases.
        if re.search(r'а[- /]*08|аренд',q):entities.add('a08')
        if re.search(r'пт[- /]*05|альфа[ -]+трейдинг',q):entities.add('pt05')
        if re.search(r'(?<!т)п[- /]*17|альфа[ -]+трейд(?:\b|[ауе]\b)',q):entities.add('p17')
        entities.intersection_update({'a08','p17','pt05'})
        ranked=[]
        for i,row in enumerate(self.rows):
            d=self.docs[row['doc_id']];entity=d.metadata.get('entity','')
            if entities and entity and entity not in entities:continue
            legal_topics={'vat_rate','vat_exemption','income_tax','profit_tax','profit_tax_special','tax_penalty','tax_sanctions','probation','fixed_term','overtime_pay','overtime_limits','minimum_wage','data_penalty','consent_requirements','consent_withdrawal'}
            if entities and d.metadata.get('domain')=='law' and not set(topics).intersection(legal_topics):continue
            if entities and d.metadata.get('type')=='memo' and 'archive' not in topics:continue
            if row['when'] and not re.search(row['when'],q):continue
            overlap=set(topics).intersection(row['topics'])
            if row['kind']=='reviewed_note' and not overlap:continue
            score=float(scores[i])
            if score<=0 and not overlap:continue
            score=max(0,score)
            score+=8*bool(overlap)+5*(entity in entities if entity else False)+3*(row['kind']=='reviewed_note')
            ranked.append((score,i,overlap))
        ranked.sort(reverse=True)
        # A place is reserved per requested topic before global lexical additions.
        order=[]
        for topic in topics:
            matches=[i for _,i,overlap in ranked if topic in overlap]
            if entities:
                for ent in sorted(entities):
                    selected=[i for i in matches if self.docs[self.rows[i]['doc_id']].metadata.get('entity')==ent]
                    order.extend(selected[:1])
            order.extend(matches[:2])
        order.extend(i for _,i,_ in ranked[:4]);order=list(dict.fromkeys(order))
        evidence=[];used=0;selected=[]
        for i in order:
            row=self.rows[i];d=self.docs[row['doc_id']]
            missing=[s for s in row['spans'] if not any(e['document']==source_name(d) and s in e['text'] for e in evidence)]
            additions=[{'source_id':f'D{len(evidence)+j+1}','document':source_name(d),'title':d.title,
                'scope':row['locator']+f', символы {d.raw.index(s)}:{d.raw.index(s)+len(s)}','text':s} for j,s in enumerate(missing)]
            cost=sum(estimate_tokens(json.dumps(e,ensure_ascii=False)) for e in additions)
            if not missing:continue
            if used+cost>budget:continue
            evidence.extend(additions);used+=cost
            selected.append({'id':row['id'],'document':d.filename,'kind':row['kind'],
                'topics':list(set(topics).intersection(row['topics'])),'source_ids':[e['source_id'] for e in additions]})
            if len(selected)>=12:break
        return evidence,{'selected':selected,'tokens':used,'available_units':len(self.rows),'valid_documents':len(self.valid)}

def combine_evidence(priority,ordinary):
    result=list(priority)
    for e in ordinary:
        if any(x['document']==e['document'] and e['text'] in x['text'] for x in result):continue
        result.append(e)
    return [{**e,'source_id':f'D{i+1}'} for i,e in enumerate(result)]
