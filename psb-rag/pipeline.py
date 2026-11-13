import json, re, threading, time
from dataclasses import asdict
from settings import Settings
from corpus import ingest
from storage import Store
from model_api import API,ModelError
from retrieval import Retriever,resolve_entity,exact_entities,ENTITY_LABELS
from schemas import Plan,Draft,Verdict,MemorySummary
from prompt_config import PROMPTS
from reasoning import parse_history,fallback_plan,evidence_payload,validate_claim,normalize_quote,PLANNER,GENERATOR,VERIFIER
from security import suspicious
from source_audit import load_audit,classify

TECHNICAL='Не удалось завершить проверку ответа из-за технического сбоя. Повторите запрос. Это не означает отсутствия сведений в базе.'

class Pipeline:
    def __init__(self,settings=None,api=None):
        self.s=settings or Settings(); self.api=api or API(self.s)
        docs,chunks,relations=ingest(self.s.data_dir)
        self.source_audit=load_audit(chunks,self.s.cache_dir)
        self.docs={d.id:d for d in docs}; self.chunks={c.id:c for c in chunks}
        self.store=Store(self.s.cache_dir/'rag.sqlite3'); self.store.sync_corpus(docs,chunks,relations)
        self.retriever=Retriever(docs,chunks,relations,self.api,self.s)
        self.log_lock=threading.Lock()
    def initialize(self):
        if self.source_audit is None:
            try:self.source_audit,_=classify(list(self.chunks.values()),self.api,self.s.cache_dir)
            except ModelError:self.source_audit={'degraded':'Classifier unavailable; trust boundary and per-answer verification remain active'}
        if self.s.retrieval_mode!='bm25':
            try: self.retriever.prepare_dense()
            except ModelError: pass
    def close(self):
        self.store.close()
        if hasattr(self.api,'close'):self.api.close()
    def ask(self,question,session_id='',mode='general'):
        if not question.strip(): return {'answer':'Укажите вопрос.','status':'clarification','sources':[],'trace':{}}
        if len(question)>24000: return {'answer':'Сократите вопрос до 24000 символов.','status':'clarification','sources':[],'trace':{}}
        if session_id:
            with self.store.session_lock(session_id):
                self.store.purge_sessions(self.s.session_ttl)
                session=self.store.session(session_id,self.s.session_ttl)
                result=self._answer(question,session,mode)
                if result['status']!='technical_error':
                    slots=result['trace'].get('slots',{})
                    if result['trace'].get('topic_changed'): session={'turns':[],'summary':'','slots':{}}
                    summary=None
                    if len(session['turns'])>=3:
                        try:
                            summary=self.api.structured(PROMPTS['memory_summary'],{'old_summary':session['summary'],'old_user_question':session['turns'][0]['user']},MemorySummary).summary
                        except (ModelError,AttributeError):result['trace']['degraded'].append('memory summary fallback')
                    self.store.save_session(session_id,session,question,result['answer'],slots,summary)
                return result
        return self._answer(question,{'turns':[],'summary':'','slots':{}},mode)
    def _answer(self,question,session,mode):
        started=time.monotonic(); latest,inline_history=parse_history(question)
        history=inline_history or session['turns']
        trace={'security_flag':suspicious(question),'source_audit':self.source_audit,'degraded':[]}
        try:
            plan=self.api.structured(PLANNER,{'question':latest,'history':history,'summary':session['summary'],
                'slots':session['slots'],'current_date':self.s.as_of},Plan)
        except ModelError as e:
            plan=fallback_plan(latest,history,self.s.as_of); trace['degraded'].append(str(e))
        # Planner must preserve explicit identifiers; a model rewrite cannot switch contract.
        explicit,candidates=resolve_entity(latest)
        planned,_=resolve_entity(plan.question)
        if explicit and planned!=explicit: plan.question=latest
        if set(exact_entities(latest))-set(exact_entities(plan.question)):plan.question=latest
        if set(map(int,re.findall(r'\d+',latest)))-set(map(int,re.findall(r'\d+',plan.question))):plan.question=latest
        if re.search(r'\bне\b',latest,re.I) and not re.search(r'\bне\b',plan.question,re.I):plan.question=latest
        if plan.topic_changed:
            plan.question=latest; history=[]
        planned,_=resolve_entity(plan.question)
        if not plan.question.strip(): plan.question=latest
        if plan.date_role!='unspecified' and plan.as_of[:4] not in latest+' '.join(x.get('user','') for x in history):
            plan.as_of=self.s.as_of;plan.date_role='unspecified'
        trace['security_flag']=trace['security_flag'] or plan.input_attack
        trace.update(plan=plan.model_dump(),topic_changed=plan.topic_changed,
            slots={'active_doc':explicit or planned,'counterparty':explicit or planned,'as_of':plan.as_of or self.s.as_of,'domain':plan.domain,'topic':plan.topic})
        if plan.unsafe_request:
            return self.finish('Не могу помогать скрывать нарушения или обходить требования закона. Могу разъяснить применимые правила по предоставленным документам.','out_of_domain',[],trace,started)
        # Genuine unresolved short name: no fuzzy forced choice.
        if len(candidates)>1 and not exact_entities(latest) and not re.search(r'сравн|обоих|оба|разниц',latest,re.I):
            return self.finish('Уточните договор: '+' или '.join(ENTITY_LABELS[c] for c in candidates)+'?','clarification',[],trace,started)
        try:
            chunks,retrieval=self.retriever.search(plan,original_query=latest); trace['retrieval']=retrieval
            evidence,budget=evidence_payload(chunks,self.docs,self.s.context_tokens)
            trace['budget']=budget
            payload={'question':latest,'standalone_question':plan.question,'plan':plan.model_dump(),
                     'evidence':evidence,'mode':mode,'current_date':self.s.as_of,
                     'inventory':[d.filename for d in self.docs.values()]}
            payload['context_limits']=budget
            payload['known_corpus_gaps']=[{'document':d.filename,'gaps':d.metadata['content_gaps']} for d in self.docs.values() if d.metadata['content_gaps']]
            draft=self.api.structured(GENERATOR,payload,Draft)
            byid={x['chunk_id']:x for x in evidence}
            accepted=[]; issues=[]; verdict=None
            for attempt in range(2):
                local={i:validate_claim(c,byid) for i,c in enumerate(draft.claims)}
                if draft.claims or draft.missing:
                    verdict=self.api.structured(VERIFIER,{**payload,'draft':draft.model_dump(),'local_issues':local},Verdict)
                    supported=set(verdict.supported_indices)
                    if any(type(i)!=int or i<0 or i>=len(draft.claims) for i in supported): raise ModelError('invalid verifier indices')
                    if any(i<0 or i>=len(draft.missing) for i in verdict.supported_missing_indices):raise ModelError('invalid missing indices')
                    expected=len(plan.subquestions) or 1
                    if len(verdict.coverage)!=expected: raise ModelError('invalid coverage shape')
                    accepted=[c for i,c in enumerate(draft.claims) if i in supported and not local[i]]
                    issues=[issue for v in local.values() for issue in v]+verdict.issues
                else:
                    accepted=[]
                if len(accepted)==len(draft.claims) and (not verdict or all(verdict.coverage)): break
                if attempt==0:
                    draft=self.api.structured(GENERATOR+'\nИсправь только обнаруженные ошибки. Удали недоказанные выводы.',
                        {**payload,'previous_draft':draft.model_dump(),'issues':issues,
                         'local_issues_by_claim':{str(i):errors for i,errors in local.items() if errors}},Draft)
            trace['verification']={'accepted':len(accepted),'proposed':len(draft.claims),'issues':issues,
                                   'coverage':verdict.coverage if verdict else [],
                                   'proposed_warnings':draft.warnings,'proposed_conflicts':draft.conflicts,
                                   'rejected_local':[{'claim':c.model_dump(),'issues':local.get(i,[])} for i,c in enumerate(draft.claims) if local.get(i)]}
            # Free-form gap text is not allowed to smuggle new legal conclusions into the answer.
            parts=[]; sources=[]
            for claim in accepted:
                citations=[]
                for cit in claim.citations:
                    e=byid[cit.chunk_id]
                    locator=e['locator']
                    if any(normalize_quote(cit.quote) in normalize_quote(note['text']) for note in e['annotations']):
                        locator='редакционная пометка «'+cit.quote[:110].strip()+('…' if len(cit.quote)>110 else '')+'»'
                    if locator=='преамбула':
                        # Unnumbered provisions get an exact textual anchor, not invented numbering.
                        locator='фрагмент «'+cit.quote[:110].strip()+('…' if len(cit.quote)>110 else '')+'»'
                    label=f'{e["document"]}, {locator}'
                    citations.append(label)
                    sources.append({**e,'quote':cit.quote})
                parts.append(claim.text.strip()+' ('+'; '.join(dict.fromkeys(citations))+').')
            status=draft.status
            gaps=[draft.missing[i] for i in verdict.supported_missing_indices] if verdict else []
            incomplete=verdict is not None and not all(verdict.coverage)
            if parts:
                if incomplete or gaps:
                    parts.append('Не удалось подтвердить: '+'; '.join(gaps) if gaps else 'Часть вопроса не удалось подтвердить предоставленными документами. Для неё нужны дополнительные сведения или отсутствующие документы.')
                    status='partial'
                else: status='answered'
            elif draft.status=='clarification':
                # Keep clarification as a question, not an unsupported legal answer.
                parts=['Уточните договор, дату события и обстоятельства, о которых идёт речь.']
            elif draft.status=='out_of_domain': parts=['Могу помочь с правовым вопросом по предоставленной базе документов. Сформулируйте такой вопрос.']
            elif draft.claims and not gaps:
                return self.finish(TECHNICAL,'technical_error',[],trace,started)
            else:
                parts=['В предоставленной базе не удалось подтвердить: '+'; '.join(gaps) if gaps else 'В предоставленных фрагментах базы нет достаточных сведений для подтверждённого ответа. Уточните обстоятельства или предоставьте относящийся к вопросу документ.']
                status='not_in_base'
            # Never truncate a citation halfway: keep whole claims within submission bound.
            answer=''; kept=[]
            for part in parts:
                if len(answer)+len(part)+2>9900: status='partial'; break
                kept.append(part); answer='\n\n'.join(kept)
            if not answer: return self.finish(TECHNICAL,'technical_error',[],trace,started)
            return self.finish(answer,status,sources,trace,started)
        except (ModelError,ValueError,KeyError,TypeError) as e:
            trace['error']=type(e).__name__+': '+str(e) if isinstance(e,ModelError) else type(e).__name__
            return self.finish(TECHNICAL,'technical_error',[],trace,started)
    def finish(self,answer,status,sources,trace,start):
        trace['seconds']=round(time.monotonic()-start,4)
        result={'answer':answer,'status':status,'sources':sources,'not_in_base':status=='not_in_base',
                'as_of':trace.get('plan',{}).get('as_of') or self.s.as_of,
                'warnings':['Untrusted instruction text detected'] if trace.get('security_flag') or any(s.get('instruction_texts') for s in sources) else [],'trace':trace}
        # No user text, source text, credentials, raw HTTP exceptions or model prompts in logs.
        log={'status':status,'seconds':trace['seconds'],'retrieval':trace.get('retrieval',{}),
             'accepted':trace.get('verification',{}).get('accepted'), 'error':trace.get('error','')}
        with self.log_lock:
            with (self.s.cache_dir/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(log,ensure_ascii=False)+'\n')
        return result
