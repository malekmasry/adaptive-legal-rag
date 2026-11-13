import json, re, threading, time
from dataclasses import asdict
from settings import Settings
from corpus import ingest
from storage import Store
from model_api import API,ModelError
from retrieval import Retriever,resolve_entity,exact_entities,ENTITY_LABELS,explicit_event_date
from schemas import Plan,Draft,Verdict,MemorySummary,Claim
from prompt_config import PROMPTS
from reasoning import parse_history,fallback_plan,evidence_payload,validate_claim,normalize_quote,PLANNER,GENERATOR,VERIFIER
from security import suspicious
from source_audit import load_audit,classify
from diagnostics import emit,error_info

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
            except ModelError as exc:
                self.source_audit={'degraded':'Classifier unavailable; source trust boundary remains active'}
                emit(self.s.cache_dir,'initialization_degraded',component='source_classifier',**error_info(exc))
        if self.s.retrieval_mode!='bm25':
            try: self.retriever.prepare_dense()
            except ModelError as exc:emit(self.s.cache_dir,'initialization_degraded',component='embeddings',**error_info(exc))
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
            emit(self.s.cache_dir,'planner_fallback',**error_info(e))
        plan.subquestions=list(dict.fromkeys(q.strip() for q in plan.subquestions if q.strip()))
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
        explicit_date=explicit_event_date(latest)
        if explicit_date:
            plan.as_of=explicit_date;plan.date_role='event'
        trace['security_flag']=trace['security_flag'] or plan.input_attack
        trace.update(plan=plan.model_dump(),topic_changed=plan.topic_changed,
            slots={'active_doc':explicit or planned,'counterparty':explicit or planned,'as_of':plan.as_of or self.s.as_of,'domain':plan.domain,'topic':plan.topic})
        if plan.unsafe_request:
            return self.finish('Не могу помогать скрывать нарушения или обходить требования закона. Могу разъяснить применимые правила по предоставленным документам.','out_of_domain',[],trace,started)
        # Genuine unresolved short name: no fuzzy forced choice.
        if len(candidates)>1 and not exact_entities(latest) and not re.search(r'сравн|обоих|оба|разниц',latest,re.I):
            return self.finish('Уточните договор: '+' или '.join(ENTITY_LABELS[c] for c in candidates)+'?','clarification',[],trace,started)
        try:
            trace['stage']='retrieval'
            chunks,retrieval=self.retriever.search(plan,original_query=latest); trace['retrieval']=retrieval
            evidence,budget=evidence_payload(chunks,self.docs,self.s.context_tokens)
            trace['budget']=budget
            payload={'question':latest,'standalone_question':plan.question,'plan':plan.model_dump(),
                     'evidence':evidence,'mode':mode,'current_date':self.s.as_of,
                     'inventory':[d.filename for d in self.docs.values()]}
            payload['context_limits']=budget
            payload['coverage_questions']=[{'index':i,'question':q} for i,q in enumerate(plan.subquestions or [plan.question])]
            payload['known_corpus_gaps']=[{'document':d.filename,'gaps':d.metadata['content_gaps']} for d in self.docs.values() if d.metadata['content_gaps']]
            trace['stage']='generation'
            draft=self.api.structured(GENERATOR,payload,Draft)
            byid={x['chunk_id']:x for x in evidence}
            accepted=[]; issues=[]; verdict=None; best=None; verification_failed=False; coverage_malformed=False
            for attempt in range(2):
                local={i:validate_claim(c,byid) for i,c in enumerate(draft.claims)}
                if draft.claims or draft.missing:
                    trace['stage']='verification'
                    try:
                        verdict=self.api.structured(VERIFIER,{**payload,'draft':draft.model_dump(),'local_issues':local},Verdict)
                    except ModelError as exc:
                        self.record_failure(trace,'verification',exc)
                        verification_failed=True
                        break
                    supported={i for i in verdict.supported_indices if type(i)==int and 0<=i<len(draft.claims)}
                    valid_missing=[i for i in dict.fromkeys(verdict.supported_missing_indices) if type(i)==int and 0<=i<len(draft.missing)]
                    if len(supported)!=len(set(verdict.supported_indices)) or len(valid_missing)!=len(set(verdict.supported_missing_indices)):
                        trace['degraded'].append('invalid verifier indices ignored')
                        emit(self.s.cache_dir,'verification_metadata_invalid',reason='invalid_indices')
                    verdict.supported_missing_indices=valid_missing
                    expected=len(plan.subquestions) or 1
                    coverage_malformed=len(verdict.coverage)!=expected
                    if coverage_malformed:
                        emit(self.s.cache_dir,'verification_metadata_invalid',reason='coverage_shape',expected=expected,received=len(verdict.coverage))
                        trace['degraded'].append('coverage shape mismatch; completeness unknown')
                        # Do not infer a positional mapping from a malformed list.
                        verdict.coverage=[False]*expected
                    accepted=[c for i,c in enumerate(draft.claims) if i in supported and not local[i]]
                    issues=[issue for v in local.values() for issue in v]+verdict.issues
                    # A failed correction must not erase already verified claims.
                    if accepted and (best is None or len(accepted)>=len(best['accepted'])):
                        best={'draft':draft,'accepted':accepted,'verdict':verdict,'local':local,'issues':issues}
                else:
                    accepted=[]
                if len(accepted)==len(draft.claims) and (not verdict or all(verdict.coverage) or coverage_malformed): break
                if attempt==0:
                    trace['stage']='correction'
                    try:
                        draft=self.api.structured(GENERATOR+'\nИсправь только обнаруженные ошибки. Сохрани подтверждённые утверждения. Удали недоказанные выводы.',
                            {**payload,'previous_draft':draft.model_dump(),'issues':issues,
                             'coverage':verdict.coverage if verdict else [],
                             'local_issues_by_claim':{str(i):errors for i,errors in local.items() if errors}},Draft)
                    except ModelError as exc:
                        self.record_failure(trace,'correction',exc)
                        verification_failed=True
                        break
            if best and (verification_failed or len(accepted)<len(best['accepted'])):
                draft,accepted,verdict,local,issues=(best[k] for k in ('draft','accepted','verdict','local','issues'))
                trace['degraded'].append('preserved verified claims from previous attempt')
            elif verification_failed:
                # Current draft has not received a semantic verdict. Never carry
                # indices from a previous draft over to new claims or gaps.
                accepted=[];verdict=None
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
            incomplete=(verification_failed and verdict is None) or (verdict is not None and not all(verdict.coverage))
            if accepted and verdict is not None and all(verdict.coverage):
                # Completeness and missing-information metadata must agree.
                # Conditions affecting the legal conclusion remain inside claims.
                trace['suppressed_gap_count']=len(gaps)
                gaps=[]
            if parts:
                if incomplete or gaps:
                    parts.append('Не удалось подтвердить: '+'; '.join(gaps) if gaps else 'Полноту ответа на все части вопроса подтвердить не удалось.')
                    status='partial'
                else: status='answered'
            elif draft.status=='clarification':
                # Keep clarification as a question, not an unsupported legal answer.
                parts=['Уточните договор, дату события и обстоятельства, о которых идёт речь.']
            elif draft.status=='out_of_domain': parts=['Могу помочь с правовым вопросом по предоставленной базе документов. Сформулируйте такой вопрос.']
            elif draft.claims and not gaps:
                parts,sources=self.quote_fallback(draft,byid)
                trace['fallback_reason']='verification_unavailable' if verification_failed else 'claims_not_supported'
                if parts:
                    parts.append('Приведены найденные положения. Итоговый вывод по вопросу подтвердить не удалось.')
                    status='partial'
                else:
                    parts=['По найденным фрагментам не удалось подтвердить вывод по вопросу. Это не означает отсутствия нужных сведений во всей базе.']
                    status='unverified'
            else:
                parts=['В предоставленной базе не удалось подтвердить: '+'; '.join(gaps) if gaps else 'В предоставленных фрагментах базы нет достаточных сведений для подтверждённого ответа. Уточните обстоятельства или предоставьте относящийся к вопросу документ.']
                status='not_in_base'
            # Never truncate a citation halfway: keep whole claims within submission bound.
            answer=''; kept=[]
            for part in parts:
                if len(answer)+len(part)+2>9700:
                    status='partial';trace['answer_length_limited']=True;continue
                kept.append(part); answer='\n\n'.join(kept)
            if not answer:
                excerpts,sources=self.quote_fallback(draft,byid)
                answer='\n\n'.join(excerpts) or 'Не удалось сформировать подтверждённый вывод в пределах допустимой длины ответа.'
                status='partial' if excerpts else 'unverified'
            if trace.get('answer_length_limited'):
                answer+='\n\nЧасть ответа не включена из-за ограничения длины.'
            trace['stage']='completed'
            return self.finish(answer,status,sources,trace,started)
        except (ModelError,ValueError,KeyError,TypeError) as e:
            self.record_failure(trace,trace.get('stage','answer'),e)
            return self.finish(TECHNICAL,'technical_error',[],trace,started)
    def record_failure(self,trace,stage,exc):
        details=error_info(exc)
        trace.setdefault('failures',[]).append({'stage':stage,**details})
        trace['error']=details['error_code']
        emit(self.s.cache_dir,'pipeline_failure',failed_stage=stage,**details)
    def quote_fallback(self,draft,byid):
        """Expose exact legal excerpts, never an unverified model conclusion."""
        parts=[];sources=[];seen=set()
        for claim in draft.claims:
            # The quote must be real and outside detected instruction spans;
            # calculations and semantic conclusions are deliberately not rendered.
            for cit in claim.citations:
                quoted=Claim(text='excerpt',citations=[cit])
                if validate_claim(quoted,byid) or (cit.chunk_id,cit.quote) in seen:continue
                seen.add((cit.chunk_id,cit.quote));e=byid[cit.chunk_id]
                quote=cit.quote.strip();short=len(quote)>900
                if short:quote=quote[:900].rsplit(' ',1)[0]
                parts.append('В документе «'+e['document']+'», '+e['locator']+', указано: «'+quote+('…' if short else '')+'».'+(' Цитата приведена частично.' if short else ''))
                sources.append({**e,'quote':quote})
                if len(parts)>=3:return parts,sources
        return parts,sources
    def finish(self,answer,status,sources,trace,start):
        trace['seconds']=round(time.monotonic()-start,4)
        result={'answer':answer,'status':status,'sources':sources,'not_in_base':status=='not_in_base',
                'as_of':trace.get('plan',{}).get('as_of') or self.s.as_of,
                'warnings':['Untrusted instruction text detected'] if trace.get('security_flag') or any(s.get('instruction_texts') for s in sources) else [],'trace':trace}
        # No user text, source text, credentials, raw HTTP exceptions or model prompts in logs.
        log={'status':status,'seconds':trace['seconds'],'retrieval':trace.get('retrieval',{}),
             'accepted':trace.get('verification',{}).get('accepted'), 'error':trace.get('error','')}
        emit(self.s.cache_dir,'answer_completed',status=status,seconds=trace['seconds'],
             accepted=log['accepted'],proposed=trace.get('verification',{}).get('proposed'),
             coverage=trace.get('verification',{}).get('coverage',[]),
             error_code=trace.get('error'),fallback_reason=trace.get('fallback_reason'),
             degraded=bool(trace.get('degraded')) or bool(trace.get('failures')),
             answer_chars=len(answer),context_tokens=trace.get('budget',{}).get('estimated_tokens_with_margin'))
        try:
            with self.log_lock:
                with (self.s.cache_dir/'events.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(log,ensure_ascii=False)+'\n')
        except OSError as exc:emit(self.s.cache_dir,'event_log_write_failed',**error_info(exc))
        return result
