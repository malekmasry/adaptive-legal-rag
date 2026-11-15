"""Role-bound paid calls. No discovery, cross-role fallback or model substitution."""
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import httpx

DEEPSEEK = 'deepseek-v4.1-flash'
QWEN = 'Qwen/Qwen3-Reranker-8B'


def validate(config):
    for role, model in [('GENERATOR',DEEPSEEK),('JUDGE',DEEPSEEK),('RERANKER',QWEN)]:
        if config.get(role+'_MODEL') != model:
            raise ValueError('Wrong model for '+role)
        for suffix in ('_BASE_URL','_API_KEY'):
            if not isinstance(config.get(role+suffix),str) or not config[role+suffix].strip():
                raise ValueError('Missing '+role+suffix)
        url=urlsplit(config[role+'_BASE_URL'])
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid endpoint for '+role)
    for suffix in ('_BASE_URL','_API_KEY'):
        if config['GENERATOR'+suffix] != config['JUDGE'+suffix]:
            raise ValueError('Generator and judge must use the same user DeepSeek account')
        if config['GENERATOR'+suffix] == config['RERANKER'+suffix]:
            raise ValueError('Qwen must use its separate endpoint and credential')
    return config


class BudgetExceeded(RuntimeError): pass


class Services:
    def __init__(self, config, directory, *, transport=None, max_chat_calls=100,
                 max_rerank_calls=8, max_chat_tokens=400000):
        self.config=validate(config); self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.client=httpx.Client(transport=transport,timeout=90,follow_redirects=False,trust_env=False)
        self.lock=threading.Lock()
        self.limits={'chat':max_chat_calls,'rerank':max_rerank_calls}
        self.token_limit=max_chat_tokens
        self.counts={'chat':0,'rerank':0};self.reserved=0
        self.ledger=self.directory/'calls.jsonl'
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf8').splitlines():
                row=json.loads(line)
                if row['event']=='reserved':
                    self.counts[row['kind']]+=1;self.reserved+=row['reserved_tokens']

    def close(self):self.client.close()

    def _append(self,value):
        with self.ledger.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False)+'\n')

    def post(self,role,payload):
        if role not in ('GENERATOR','JUDGE','RERANKER'):raise ValueError('Forbidden API role')
        expected=self.config[role+'_MODEL']
        if payload.get('model')!=expected:raise ValueError('Wrong model blocked before HTTP')
        kind='rerank' if role=='RERANKER' else 'chat'
        path='/rerank' if kind=='rerank' else '/chat/completions'
        raw=json.dumps(payload,ensure_ascii=False)
        # UTF-8 bytes are a conservative upper bound for common byte-level tokenizers.
        reservation=len(raw.encode('utf8'))+payload.get('max_tokens',0) if kind=='chat' else 0
        with self.lock:
            if self.counts[kind]>=self.limits[kind] or self.reserved+reservation>self.token_limit:
                raise BudgetExceeded('Frozen API budget reached; cached work retained')
            self.counts[kind]+=1;self.reserved+=reservation
            call=f'{kind}-{self.counts[kind]}'
            self._append({'event':'reserved','call':call,'role':role,'kind':kind,'model':expected,
                          'reserved_tokens':reservation,'payload_sha256':hashlib.sha256(raw.encode()).hexdigest()})
        start=time.monotonic();record={'event':'completed','call':call,'role':role,'kind':kind,'model':expected}
        try:
            response=self.client.post(self.config[role+'_BASE_URL'].rstrip('/')+path,json=payload,
                headers={'Authorization':'Bearer '+self.config[role+'_API_KEY']})
            record['http_status']=response.status_code
            if response.status_code!=200:raise RuntimeError('API HTTP '+str(response.status_code))
            result=response.json()
            if result.get('model') not in (None,expected):raise RuntimeError('Unexpected returned model')
            record['usage']=result.get('usage',{})
            if kind=='rerank':record['meta']=result.get('meta',{})
            record['status']='ok'
            return result
        except Exception as exc:
            record.update(status='error',error_type=type(exc).__name__)
            raise
        finally:
            record['seconds']=round(time.monotonic()-start,3)
            with self.lock:self._append(record)

    def chat(self,role,prompt,payload,max_tokens=650):
        if role not in ('GENERATOR','JUDGE'):raise ValueError('Reranker cannot generate text')
        result=self.post(role,{'model':DEEPSEEK,'temperature':0,'max_tokens':max_tokens,
            'response_format':{'type':'json_object'},'messages':[{'role':'system','content':prompt},
            {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]})
        choice=result['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('Incomplete model response')
        value=json.loads(choice['message']['content'])
        if not isinstance(value,dict):raise ValueError('Expected object')
        return value

    def rerank(self,query,documents,top_n=6):
        result=self.post('RERANKER',{'model':QWEN,'query':query,'documents':documents,'top_n':min(top_n,len(documents))})
        rows=result.get('results',result.get('data',[]));indices=[r['index'] for r in rows]
        if not indices or len(indices)!=len(set(indices)) or any(type(i)!=int or not 0<=i<len(documents) for i in indices):
            raise ValueError('Invalid Qwen ranking')
        if len(indices)>min(top_n,len(documents)):raise ValueError('Too many reranker results')
        return indices
