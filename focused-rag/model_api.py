import json, math, threading, time
import httpx
import numpy as np
from pydantic import ValidationError
from security import TRUST
from token_budget import estimate_tokens
from diagnostics import emit, error_info, stage_scope
from schemas import CompactPlan

def schema_text(schema):
    value=schema.model_json_schema()
    if schema is CompactPlan:
        def prune(node):
            if isinstance(node,dict):return {k:prune(v) for k,v in node.items() if k!='title'}
            if isinstance(node,list):return [prune(v) for v in node]
            return node
        return json.dumps(prune(value),ensure_ascii=False,separators=(',',':'))
    return json.dumps(value,ensure_ascii=False)

class ModelError(RuntimeError):
    def __init__(self,message,code=None,details=None):
        super().__init__(message)
        self.code=code or {
            'Run deadline reached':'run_deadline',
            'chat schema validation failed':'json_schema_invalid',
            'chat output truncated: finish_reason=length':'output_truncated',
            'LLM_MODEL missing':'generator_model_missing',
            'embedding contract violation':'embedding_contract_invalid',
            'rerank contract violation':'rerank_contract_invalid',
            'query embedding dimension mismatch':'embedding_dimension_mismatch',
            'Incomplete instruction classification':'source_classification_incomplete',
        }.get(message,'model_error')
        self.details=details or []

class API:
    def __init__(self,settings,transport=None):
        self.s=settings
        self.client=httpx.Client(transport=transport,limits=httpx.Limits(max_connections=16,max_keepalive_connections=8))
        self.events=[]; self.lock=threading.Lock()
        self.deadline=time.monotonic()+settings.deadline_seconds
    def close(self): self.client.close()
    def post(self,path,payload,kind):
        s=self.s
        base=(s.emb_url if kind=='embedding' else s.rerank_url if kind=='rerank' else '') or s.base_url
        key=(s.emb_key if kind=='embedding' else s.rerank_key if kind=='rerank' else '') or s.api_key
        if not base or not key: raise ModelError('API configuration missing: '+kind,code='api_configuration_missing')
        start=time.monotonic(); status='failed';usage={};http_status=None;finish_reason=None
        try:
            for attempt in range(3):
                remaining=self.deadline-time.monotonic()
                if remaining<=1: raise ModelError('Run deadline reached')
                try:
                    r=self.client.post(base.rstrip('/')+path,json=payload,
                        headers={'Authorization':'Bearer '+key},timeout=httpx.Timeout(min(s.request_timeout,remaining),connect=min(8,remaining)))
                    http_status=r.status_code
                    if r.status_code in (408,429,500,502,503,504) and attempt<2:
                        emit(s.cache_dir,'api_retry',kind=kind,http_status=r.status_code,attempt=attempt+1)
                        delay=min(2**attempt,float(r.headers.get('retry-after','1')) if r.headers.get('retry-after','1').isdigit() else 1, max(0,self.deadline-time.monotonic()))
                        time.sleep(delay); continue
                    if r.is_error: raise ModelError(f'{kind} HTTP {r.status_code}',code='http_'+str(r.status_code))
                    result=r.json(); status='ok'
                    if isinstance(result,dict) and isinstance(result.get('usage'),dict):
                        usage={k:v for k,v in result['usage'].items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int)}
                    if kind=='chat' and isinstance(result,dict):
                        choices=result.get('choices')
                        if isinstance(choices,list) and choices and isinstance(choices[0],dict):
                            reason=choices[0].get('finish_reason')
                            finish_reason=reason if reason in ('stop','length','content_filter','tool_calls') else 'unknown'
                    return result
                except httpx.TransportError as exc:
                    emit(s.cache_dir,'transport_failure',kind=kind,attempt=attempt+1,error_type=type(exc).__name__)
                    if attempt==2: raise ModelError(kind+' transport failure',code='transport_failure') from None
            raise ModelError(kind+' retries exhausted')
        except (ValueError,KeyError,TypeError): raise ModelError(kind+' malformed response',code='api_response_invalid') from None
        finally:
            event={'kind':kind,'model':payload.get('model',''),'seconds':round(time.monotonic()-start,4),'status':status,'usage':usage,
                'http_status':http_status,'finish_reason':finish_reason,'max_tokens':payload.get('max_tokens'),
                'estimated_prompt_tokens':estimate_tokens(json.dumps(payload.get('messages',[]),ensure_ascii=False),margin=1) if kind=='chat' else None}
            with self.lock: self.events.append(event)
            emit(s.cache_dir,'api_call',**event)
    def structured(self,instruction,payload,schema):
        # One bounded retry for formatting or truncation; do not silently override
        # the platform's configured MAX_TOKENS or retry permanent HTTP failures.
        with stage_scope(schema.__name__):
            for attempt in range(2):
                try:
                    return self._structured_once(instruction,payload,schema)
                except ModelError as exc:
                    emit(self.s.cache_dir,'structured_failure',schema=schema.__name__,attempt=attempt+1,
                         validation=exc.details,**error_info(exc))
                    if attempt or exc.code not in ('json_schema_invalid','output_truncated','api_response_invalid'):
                        raise
                    instruction+='\nПовтори полный валидный JSON по схеме. Только обязательные и содержательные поля. '
                    if exc.code=='output_truncated':
                        instruction+='Предыдущий ответ оборвался по лимиту. Сократи цитаты до достаточных непрерывных фрагментов, не дублируй claims в answer, не повторяй evidence. Сохрани условия и исключения.'
                    else:
                        instruction+='Исправь типы и обязательные поля. Ошибки структуры: '+json.dumps(exc.details,ensure_ascii=False)
    def _structured_once(self,instruction,payload,schema):
        if not self.s.model: raise ModelError('LLM_MODEL missing')
        data=self.post('/chat/completions',{'model':self.s.model,'temperature':self.s.temperature,
            'max_tokens':self.s.max_tokens,'response_format':{'type':'json_object'},
            'messages':[{'role':'system','content':TRUST+'\n'+instruction+'\nJSON schema:\n'+schema_text(schema)},
                        {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]},'chat')
        try:
            if data['choices'][0].get('finish_reason')=='length': raise ModelError('chat output truncated: finish_reason=length')
            content=data['choices'][0]['message']['content']
            if not isinstance(content,str):raise ValueError('missing text')
            content=content.strip()
            if content.startswith('```'): content=content.split('\n',1)[1].rsplit('```',1)[0]
            # Accept a JSON object surrounded by a model's prose; never guess
            # missing facts, close truncated JSON, or evaluate Python literals.
            try:value=json.loads(content)
            except json.JSONDecodeError:
                begin=content.find('{')
                if begin<0:raise
                value,end=json.JSONDecoder().raw_decode(content[begin:])
                if '{' in content[begin+end:]:raise ValueError('multiple objects')
            return schema.model_validate(value)
        except ValidationError as exc:
            details=[{'path':'.'.join(map(str,e['loc'])),'type':e['type']} for e in exc.errors(include_input=False,include_url=False)[:12]]
            raise ModelError('chat schema validation failed',details=details) from None
        except (ValueError,KeyError,TypeError,IndexError,AttributeError):
            raise ModelError('chat schema validation failed',details=[{'type':'invalid_json_envelope'}]) from None
    def embed(self,texts):
        if not texts: return np.empty((0,0),dtype=np.float32)
        vectors=[]; dim=None
        for start in range(0,len(texts),24):
            batch=texts[start:start+24]
            result=self.post('/embeddings',{'model':self.s.emb_model,'input':batch,'encoding_format':'float'},'embedding')
            try:
                entries=result['data']
                indices=[x['index'] for x in entries]
                if len(indices)!=len(batch) or set(indices)!=set(range(len(batch))): raise ValueError()
                matrix=np.asarray([x['embedding'] for x in sorted(entries,key=lambda x:x['index'])],dtype=np.float32)
                if matrix.ndim!=2 or not matrix.shape[1] or not np.isfinite(matrix).all(): raise ValueError()
                lengths=np.linalg.norm(matrix,axis=1)
                if (lengths<=0).any() or (dim is not None and dim!=matrix.shape[1]): raise ValueError()
                dim=matrix.shape[1]; vectors.extend(matrix/lengths[:,None])
            except (ValueError,KeyError,TypeError): raise ModelError('embedding contract violation') from None
        return np.asarray(vectors,dtype=np.float32)
    def rerank(self,query,texts,top_n):
        if not texts: return []
        n=min(top_n,len(texts))
        # PSB documents only query/documents, not a separate instruction parameter.
        # Carry the task instruction as query text; native server template is not assumed.
        instructed='<Instruct>: '+self.s.rerank_instruction+'\n<Query>: '+json.dumps(query,ensure_ascii=False)
        data=self.post('/rerank',{'model':self.s.rerank_model,'query':instructed,'documents':texts,'top_n':n},'rerank')
        try:
            results=data.get('results',data.get('data'))
            rows=[(x['index'],float(x['relevance_score'])) for x in results]
            if not rows or len(rows)>n or len({i for i,_ in rows})!=len(rows): raise ValueError()
            if any(type(i)!=int or i<0 or i>=len(texts) or not math.isfinite(score) for i,score in rows): raise ValueError()
            return sorted(rows,key=lambda x:-x[1])
        except (ValueError,KeyError,TypeError): raise ModelError('rerank contract violation') from None
