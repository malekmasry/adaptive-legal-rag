"""Runtime configuration. No embedded credentials or model substitutions."""
from dataclasses import dataclass, field
from pathlib import Path
import math, os
from datetime import date
from prompt_config import PROMPTS

def env(*names, default=""):
    for name in names:
        value=os.environ.get(name,'').strip()
        if value and value!=name: return value
    return default

def env_int(name, default, low, high):
    try: return max(low,min(high,int(env(name))))
    except ValueError: return default

def env_float(name, default, low, high):
    try:
        value=float(env(name))
        return max(low,min(high,value)) if math.isfinite(value) else default
    except ValueError: return default

@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: env('LLM_BASE_URL','OPENAI_BASE_URL','BASE_URL'))
    api_key: str = field(default_factory=lambda: env('LLM_API_KEY','OPENAI_API_KEY'))
    model: str = field(default_factory=lambda: env('LLM_MODEL','GEN_MODEL','OPENAI_MODEL'))
    emb_model: str = field(default_factory=lambda: env('EMB_MODEL',default='BAAI/bge-m3'))
    rerank_model: str = field(default_factory=lambda: env('RERANKER_MODEL',default='Qwen/Qwen3-Reranker-8B'))
    emb_url: str = field(default_factory=lambda: env('EMBEDDER_BASE_URL'))
    rerank_url: str = field(default_factory=lambda: env('RERANKER_BASE_URL'))
    emb_key: str = field(default_factory=lambda: env('EMBEDDER_API_KEY'))
    rerank_key: str = field(default_factory=lambda: env('RERANKER_API_KEY'))
    data_dir: Path = field(default_factory=lambda: Path(env('DATA_DIR',default='data')))
    cache_dir: Path = field(default_factory=lambda: Path(env('CACHE_DIR',default='cache')))
    top_k: int = field(default_factory=lambda: env_int('TOP_K',40,1,200))
    rerank_n: int = field(default_factory=lambda: env_int('RERANKER_TOP_N',6,1,30))
    workers: int = field(default_factory=lambda: env_int('MAX_WORKERS',8,1,8))
    max_tokens: int = field(default_factory=lambda: env_int('MAX_TOKENS',8192,128,32768))
    temperature: float = field(default_factory=lambda: env_float('TEMPERATURE',0,0,2))
    context_tokens: int = field(default_factory=lambda: env_int('CONTEXT_TOKENS',18000,1024,100000))
    rerank_instruction: str = field(default_factory=lambda: env('RERANK_INSTRUCTION',default=PROMPTS['reranker']))
    request_timeout: float = field(default_factory=lambda: env_float('REQUEST_TIMEOUT',90,1,180))
    deadline_seconds: float = field(default_factory=lambda: env_float('RUN_BUDGET_SECONDS',1100,1,1100))
    session_ttl: int = field(default_factory=lambda: env_int('SESSION_TTL',3600,1,86400))
    retrieval_mode: str = field(default_factory=lambda: env('RETRIEVAL_MODE',default='hybrid'))
    use_rerank: bool = field(default_factory=lambda: env('USE_RERANK',default='1')=='1')
    as_of: str = field(default_factory=lambda: env('AS_OF',default=date.today().isoformat()))

    def __post_init__(self):
        if self.retrieval_mode not in {'hybrid','bm25'}:
            raise ValueError('Invalid RETRIEVAL_MODE')
        if not 1 <= self.rerank_n <= 30 or not 1 <= self.top_k <= 200:
            raise ValueError('Retrieval bounds invalid')
        self.cache_dir.mkdir(parents=True,exist_ok=True)
