import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from settings import Settings

class TestSettings(unittest.TestCase):
    def test_existing_platform_retrieval_names_and_spec_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,{},clear=True):
                s=Settings(cache_dir=Path(directory))
                self.assertEqual((s.top_k,s.rerank_n),(40,6))
            with patch.dict(os.environ,{'TOP_K':'35','RERANKER_TOP_N':'6'},clear=True):
                s=Settings(cache_dir=Path(directory))
                self.assertEqual((s.top_k,s.rerank_n),(35,6))

    def test_blank_or_placeholder_primary_uses_platform_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            config={'LLM_MODEL':' LLM_MODEL ','GEN_MODEL':' ','OPENAI_MODEL':' supplied-model ',
                    'LLM_BASE_URL':'LLM_BASE_URL','OPENAI_BASE_URL':' https://proxy.invalid/v1/ ',
                    'LLM_API_KEY':'LLM_API_KEY','OPENAI_API_KEY':'test-key'}
            with patch.dict(os.environ,config,clear=True):
                s=Settings(cache_dir=Path(directory))
                self.assertEqual(s.model,'supplied-model')
                self.assertEqual(s.base_url,'https://proxy.invalid/v1/')
                self.assertEqual(s.api_key,'test-key')

    def test_malformed_numeric_environment_uses_safe_defaults(self):
        mapping={'MAX_WORKERS':('workers',8),'MAX_TOKENS':('max_tokens',8192),
                 'TEMPERATURE':('temperature',0),'CONTEXT_TOKENS':('context_tokens',18000),
                 'REQUEST_TIMEOUT':('request_timeout',90),'RUN_BUDGET_SECONDS':('deadline_seconds',1100),
                 'SESSION_TTL':('session_ttl',3600),'TOP_K':('top_k',40),'RERANKER_TOP_N':('rerank_n',6)}
        with tempfile.TemporaryDirectory() as directory:
            for value in ('',' ','not-a-number','nan','inf'):
                with self.subTest(value=value),patch.dict(os.environ,{k:value for k in mapping},clear=True):
                    s=Settings(cache_dir=Path(directory))
                    for attr,expected in mapping.values():self.assertEqual(getattr(s,attr),expected)
            with patch.dict(os.environ,{'MAX_WORKERS':'100','TEMPERATURE':'-1','RUN_BUDGET_SECONDS':'99999'},clear=True):
                s=Settings(cache_dir=Path(directory))
                self.assertEqual((s.workers,s.temperature,s.deadline_seconds),(8,0,1100))

    def test_platform_chunker_does_not_select_our_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            for platform_value in ('common','structural-v1','future-platform-option',''):
                with self.subTest(platform_value=platform_value),patch.dict(os.environ,{'CHUNKER':platform_value},clear=True):
                    self.assertFalse(hasattr(Settings(cache_dir=Path(directory)),'chunker'))

    def test_removed_chunker_setting_cannot_block_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,{'CHUNKER':'common','RAG_CHUNKER':'structural-v1'},clear=True):
                self.assertFalse(hasattr(Settings(cache_dir=Path(directory)),'chunker'))
            with patch.dict(os.environ,{'CHUNKER':'common','RAG_CHUNKER':'unsupported'},clear=True):
                self.assertFalse(hasattr(Settings(cache_dir=Path(directory)),'chunker'))

if __name__=='__main__':unittest.main(verbosity=2)
