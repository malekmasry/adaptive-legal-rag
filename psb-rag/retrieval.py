import json, re, time, calendar
from datetime import date
from pathlib import Path
import numpy as np
import yaml
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from corpus import norm, digest, PARSER_VERSION, applicable
from model_api import ModelError

ALIAS_CONFIG=yaml.safe_load(Path(__file__).with_name('aliases.yaml').read_text(encoding='utf8'))
ALIASES={key:value['aliases'] for key,value in ALIAS_CONFIG['entities'].items()}
ENTITY_LABELS={key:value['label'] for key,value in ALIAS_CONFIG['entities'].items()}
if any(not isinstance(a,str) or not a.strip() for values in ALIASES.values() for a in values):raise ValueError('Invalid alias table')
if any(not set(row['entities'])<=set(ALIASES) for row in ALIAS_CONFIG['ambiguous']):raise ValueError('Unknown entity in ambiguous alias')

def exact_entities(query):
    q=norm(query)
    return [key for key,aliases in ALIASES.items() if any(re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',q) for a in aliases)]

def resolve_entity(query):
    q=norm(query)
    exact=exact_entities(query)
    if len(exact)==1: return exact[0],[]
    if len(exact)>1: return '',exact
    for row in ALIAS_CONFIG['ambiguous']:
        if re.search(row['pattern'],q):return '',row['entities']
    candidates=[k for k,v in ALIASES.items() if max(fuzz.partial_ratio(q,a) for a in v[-1:])>=85]
    return '',candidates

def controlled_queries(plan,original):
    """Rewrites cannot introduce a different explicit entity or numeric condition."""
    allowed_entities=set(exact_entities(original+' '+plan.question))
    numbers=set(re.findall(r'\d+',original+' '+plan.question))
    rewrites=[q for q in plan.rewrites if set(exact_entities(q))<=allowed_entities
              and set(re.findall(r'\d+',q))<=numbers]
    return list(dict.fromkeys(([original] if original else [])+[plan.question]+plan.subquestions+rewrites))[:8]

def tokens(text,truncate=True):
    words=re.findall(r'[а-яa-z0-9]+(?:[-/.][а-яa-z0-9]+)*',norm(text))
    stop={'а','и','в','на','по','с','со','у','к','о','об','от','для','из','или','как','какой','какая','какие','что','это','ли','же','то','за','при'}
    words=[w for w in words if w not in stop]
    return [w[:6] if truncate and len(w)>6 and w.isalpha() else w for w in words]

def rrf(lists,k=60):
    scores={}
    for ranked in lists:
        for rank,id in enumerate(dict.fromkeys(ranked),1): scores[id]=scores.get(id,0)+1/(k+rank)
    return sorted(scores,key=lambda x:(-scores[x],x)),scores

def explicit_event_date(question):
    """Explicit historical approval/event dates must survive a mistaken LLM plan.

    A contract number or signature year alone must not date a question about
    current obligations. Specification dates remain a separate condition.
    """
    q=norm(question)
    if re.search(r'\bсейчас\b|\bсегодня\b|\bтеперь\b',q):return ''
    if not re.search(r'одобр|соглас(?:ов|у)|на дату|по состоянию|действовал|требовал|нужно было|должен был',q):return ''
    dates=re.findall(r'\b(\d{2})\.(\d{2})\.(\d{4})\b',q)
    if len(dates)==1:
        day,month,year=map(int,dates[0])
        try:return date(year,month,day).isoformat()
        except ValueError:return ''
    iso=re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b',q)
    if len(iso)==1:
        try:return date.fromisoformat(iso[0]).isoformat()
        except ValueError:return ''
    months={'январ':1,'феврал':2,'март':3,'апрел':4,'ма':5,'июн':6,
            'июл':7,'август':8,'сентябр':9,'октябр':10,'ноябр':11,'декабр':12}
    found=re.findall(r'\b(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йея]|июн[ьяе]|июл[ьяе]|август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])\s+(20\d{2})\b',q)
    if len(found)==1:
        name,year=found[0]
        month=next(value for stem,value in months.items() if name.startswith(stem))
        return f'{year}-{month:02}'
    years=re.findall(r'\b(?:в|за|на)\s+(20\d{2})(?:\s*г(?:оду|од|\.)?)?\b',q)
    return years[0] if len(set(years))==1 else ''

def event_window(plan,current):
    q=norm(plan.question)
    explicit=explicit_event_date(q)
    if explicit:
        if len(explicit)==10:return explicit,explicit
        if len(explicit)==7:
            y,m=map(int,explicit.split('-'))
            return explicit+'-01',explicit+f'-{calendar.monthrange(y,m)[1]:02}'
        return explicit+'-01-01',explicit+'-12-31'
    dates=re.findall(r'\b(\d{2})\.(\d{2})\.(\d{4})\b',q)
    if len(dates)==1 and re.search(r'одобр|соглас(?:ов|у)|на дату|по состоянию',q) and not re.search(r'\bсейчас\b|\bсегодня\b|\bтеперь\b',q):
        d,m,y=dates[0]
        try:
            value=date(int(y),int(m),int(d)).isoformat()
            return value,value
        except ValueError:pass
    historical=plan.date_role=='event' or (plan.date_role=='contract_signature' and re.search(r'одобр|соглас(?:ов|у)|требовал|нужно было|должен был',q))
    if re.search(r'\bсейчас\b|\bсегодня\b|\bтеперь\b',q):historical=False
    if not historical:return current,current
    value=plan.as_of
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):date.fromisoformat(value);return value,value
        if re.fullmatch(r'\d{4}-\d{2}',value):
            y,m=map(int,value.split('-'));return value+'-01',value+f'-{calendar.monthrange(y,m)[1]:02}'
        if re.fullmatch(r'\d{4}',value):return value+'-01-01',value+'-12-31'
    except ValueError:pass
    # Unknown dates must not hard-exclude a potentially correct edition.
    return '0001-01-01','9999-12-31'

class Retriever:
    def __init__(self,docs,chunks,relations,api,settings):
        self.docs={d.id:d for d in docs}; self.all={c.id:c for c in chunks}
        self.chunks=[c for c in chunks if c.kind=='passage' and not c.locator.startswith('раздел ')]
        self.by_id={c.id:i for i,c in enumerate(self.chunks)}
        self.bm=BM25Okapi([tokens(c.text) for c in self.chunks])
        self.prefix_bm=BM25Okapi([tokens(c.prefix+' '+self.docs[c.doc_id].filename) for c in self.chunks])
        self.api=api; self.s=settings; self.relations=relations; self.vectors=None
        self.dense_error=None
        self.cache_key=digest(json.dumps({'model':settings.emb_model,'parser':PARSER_VERSION,'schema':1,
            'text':[(c.id,digest(c.search_text)) for c in self.chunks]},sort_keys=True))
        self.cache_path=settings.cache_dir/('vectors-'+self.cache_key+'.npz')
        bundled=Path(__file__).resolve().parent/self.cache_path.name
        existing=self.cache_path if self.cache_path.exists() else bundled
        if existing.exists():
            try:
                with np.load(existing,allow_pickle=False) as z:
                    a=z['vectors']
                    if list(z['ids'])!=[c.id for c in self.chunks] or a.ndim!=2 or len(a)!=len(self.chunks) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-4): raise ValueError()
                    self.vectors=a
            except (ValueError,KeyError,OSError): pass
    def prepare_dense(self):
        if self.dense_error: raise ModelError(self.dense_error)
        if self.vectors is None:
            try: a=self.api.embed([c.search_text for c in self.chunks])
            except ModelError as e:
                self.dense_error=str(e);raise
            temp=self.cache_path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,vectors=a,ids=np.array([c.id for c in self.chunks]))
            temp.replace(self.cache_path); self.vectors=a
    def pool(self,plan):
        entity,candidates=resolve_entity(plan.question)
        # Only deterministically confirmed entity/date restrict the pool. LLM router is a hint.
        begin,end=event_window(plan,self.s.as_of)
        return [i for i,c in enumerate(self.chunks)
                if (self.docs[c.doc_id].metadata['type']=='contract' or
                    ((not self.docs[c.doc_id].metadata['valid_from'] or self.docs[c.doc_id].metadata['valid_from']<=end)
                    and (not self.docs[c.doc_id].metadata['valid_to'] or self.docs[c.doc_id].metadata['valid_to']>begin)))
                and (not entity or not self.docs[c.doc_id].metadata['entity'] or self.docs[c.doc_id].metadata['entity']==entity)]
    def lexical(self,query,pool,limit=None):
        scores=self.bm.get_scores(tokens(query))+.15*self.prefix_bm.get_scores(tokens(query)); limit=limit or self.s.top_k
        return [(self.chunks[i].id,float(scores[i])) for i in sorted(pool,key=lambda x:-scores[x])[:limit] if scores[i]>0]
    def dense(self,query,pool):
        return self.dense_many([query],pool)[0]
    def dense_many(self,queries,pool):
        self.prepare_dense()
        try:vectors=np.asarray(self.api.embed(queries),dtype=np.float32)
        except (ValueError,TypeError):raise ModelError('embedding contract violation') from None
        if vectors.ndim!=2 or len(vectors)!=len(queries) or not np.isfinite(vectors).all():
            raise ModelError('embedding contract violation')
        if vectors.shape[1]!=self.vectors.shape[1]:raise ModelError('query embedding dimension mismatch')
        rankings=[]
        for v in vectors:
            # Keep the exact per-query dot product and tie order used before batching.
            scores=self.vectors@v
            rankings.append([(self.chunks[i].id,float(scores[i])) for i in sorted(pool,key=lambda x:-scores[x])[:self.s.top_k]])
        return rankings
    def search(self,plan,mode=None,original_query=''):
        start=time.monotonic(); mode=mode or self.s.retrieval_mode; pool=self.pool(plan)
        trace={'mode':mode,'pool':len(pool),'rankings':[],'degraded':[]}
        queries=controlled_queries(plan,original_query)
        dense_rankings=[None]*len(queries)
        if mode=='hybrid':
            trace['query_embedding_batch_size']=len(queries)
            trace['query_embedding_mode']='batch'
            try:dense_rankings=self.dense_many(queries,pool)
            except ModelError as exc:
                trace['degraded'].append(str(exc))
                # Do not multiply exhausted retries, auth errors or a deadline by
                # the number of query variants. Recover item-specific failures only.
                recoverable=exc.code in {'embedding_contract_invalid','api_response_invalid','http_400','http_413','http_422','model_error'}
                if len(queries)>1 and recoverable and not self.dense_error:
                    trace['query_embedding_mode']='per_query_recovery'
                    for i,q in enumerate(queries):
                        try:dense_rankings[i]=self.dense(q,pool)
                        except ModelError as item_error:trace['degraded'].append(str(item_error))
                else:trace['query_embedding_mode']='unavailable'
        ranked=[]
        for i,q in enumerate(queries):
            lex=self.lexical(q,pool)
            ranked.append([id for id,_ in lex]);trace['rankings'].append({'method':'bm25','hits':lex})
            if dense_rankings[i] is not None:
                den=dense_rankings[i];ranked.append([id for id,_ in den]);trace['rankings'].append({'method':'dense','hits':den})
        ids,scores=rrf(ranked)
        trace['fusion']='RRF of BM25 and BGE-M3 dense branches'
        ids=ids[:min(80,self.s.top_k*2)]
        entity,_=resolve_entity(plan.question)
        if entity:
            # These supplied contracts are small. Exact identity justifies completing their
            # candidate pool, so colloquial wording cannot hide an otherwise relevant clause.
            entity_ids=[self.chunks[i].id for i in pool if self.docs[self.chunks[i].doc_id].metadata['entity']==entity]
            ids=list(dict.fromkeys(ids[:40]+entity_ids+ids[40:]))[:80]
        trace['candidates']=ids.copy()
        reranked=False
        if self.s.use_rerank and ids:
            try:
                rr=self.api.rerank(plan.question,[self.all[i].search_text for i in ids],self.s.rerank_n)
                trace['rerank']=[(ids[i],s) for i,s in rr]; ids=[ids[i] for i,_ in rr];reranked=True
            except ModelError as e: trace['degraded'].append(str(e)); ids=ids[:self.s.rerank_n]
        else: ids=ids[:self.s.rerank_n]
        if not reranked and original_query:
            # In degraded mode, broad subqueries must not crowd out the exact question.
            ids=list(dict.fromkeys([i for i,_ in self.lexical(original_query,pool,2)]+ids))[:self.s.rerank_n]
        ids=list(dict.fromkeys(ids))
        # A notice about an absent provision is evidence of the gap, not of its content.
        gap_anchors=[]
        for d in self.docs.values():
            article=re.search(r'ст(\d+)',d.filename)
            if not article or not re.search(r'\b'+article[1]+r'\b',plan.question):continue
            for gap in d.metadata.get('content_gaps',[]):
                number=re.findall(r'\d+',gap['locator'])[0]
                if re.search(r'\b'+number+r'\b',plan.question):
                    gap_anchors.extend(self.chunks[i].id for i in pool if self.chunks[i].doc_id==d.id and gap['notice'] in self.chunks[i].text)
        ids=list(dict.fromkeys(gap_anchors+ids))
        # Per-subquestion evidence is retained even if the joint reranker favors one topic.
        for sub in plan.subquestions:
            ids.extend(i for i,_ in self.lexical(sub,pool,2))
        expanded=self.expand(list(dict.fromkeys(ids)),pool)
        trace.update(selected=expanded,rrf=[(i,scores.get(i,0)) for i in ids],seconds=round(time.monotonic()-start,4))
        return [self.all[i] for i in expanded],trace
    def expand(self,ids,pool):
        allowed={self.chunks[i].id for i in pool}; out=list(ids)
        selected_docs={self.all[i].doc_id for i in ids}
        for id in ids:
            c=self.all[id]
            # Parent carries list preamble and conditions; complete law clause retained.
            if c.parent_id: out.append(c.parent_id)
            neighbors=[x for x in self.chunks if x.doc_id==c.doc_id and abs(x.ordinal-c.ordinal)<=1]
            out.extend(x.id for x in neighbors if x.id in allowed)
            # КоАП closing notes qualify sanctions across multiple points.
            if 'КоАП' in self.docs[c.doc_id].filename:
                out.extend(x.id for x in self.chunks if x.doc_id==c.doc_id and 'Примечани' in x.text)
        for rel in self.relations:
            if rel['kind']=='supersedes': continue  # Never bring the inapplicable edition back.
            if rel['source'] in selected_docs or rel['target'] in selected_docs:
                other=rel['target'] if rel['source'] in selected_docs else rel['source']
                if rel['kind'] in ('amends','supplements'):
                    out.extend(x.id for x in self.chunks if x.doc_id==other and x.id in allowed)
                elif rel['kind']=='cites':
                    out.extend(x.id for x in self.chunks if x.doc_id==other and x.id in allowed and x.locator!='преамбула')
        return list(dict.fromkeys(out))
