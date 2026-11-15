import tempfile
import unittest
import httpx
from resume_checked_candidate import CircuitServices
from services import DEEPSEEK,QWEN

class RecoveryTests(unittest.TestCase):
    def test_unavailable_chat_stops_later_judge_and_reranker_http(self):
        config={}
        for role,model,host,key in [('GENERATOR',DEEPSEEK,'deepseek.test','deepseek-test'),
                                   ('JUDGE',DEEPSEEK,'deepseek.test','deepseek-test'),
                                   ('RERANKER',QWEN,'qwen.test','qwen-test')]:
            config.update({role+'_MODEL':model,role+'_BASE_URL':'https://'+host,role+'_API_KEY':key})
        calls=[]
        def handle(request):
            calls.append(request.url.path);return httpx.Response(402,json={'error':'unavailable'})
        with tempfile.TemporaryDirectory() as directory:
            api=CircuitServices(config,directory,transport=httpx.MockTransport(handle))
            try:
                for role,model in [('GENERATOR',DEEPSEEK),('JUDGE',DEEPSEEK),('RERANKER',QWEN)]:
                    with self.assertRaises(RuntimeError):api.post(role,{'model':model})
                self.assertEqual(calls,['/chat/completions'])
                self.assertEqual(api.counts,{'chat':1,'rerank':0})
            finally:api.close()

if __name__=='__main__':unittest.main()
