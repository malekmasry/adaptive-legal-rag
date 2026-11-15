import json
import tempfile
import threading
import unittest
import httpx
from services import Services,DEEPSEEK,QWEN
from test_proxy import config
from stress_runtime import make_gateway
from prepare_stress import IDS,RATIONALE


class StressTests(unittest.TestCase):
    def test_twenty_explicit_adversarial_cases(self):
        self.assertEqual(len(IDS),20);self.assertEqual(len(set(IDS)),20)
        self.assertEqual(set(IDS),set(RATIONALE))
        for pair in [('S07a','S07b'),('S14a','S14b'),('S17a','S17b')]:
            self.assertTrue(set(pair)<=set(IDS))
        self.assertTrue({'S20c','S22c','S23c','S04c'}<=set(IDS))

    def test_runtime_firewall_and_local_embeddings(self):
        seen=[]
        def upstream(request):
            data=json.loads(request.content)
            seen.append((request.url.host,request.url.path,data['model']))
            return httpx.Response(200,json={'model':data['model'],'ok':True})
        class Embeddings:
            def embed(self,texts):return [[.1,.2,.3] for _ in texts],0
        with tempfile.TemporaryDirectory() as directory:
            api=Services(config(),directory,transport=httpx.MockTransport(upstream))
            server=make_gateway(api,Embeddings())
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}/v1',trust_env=False,
                                  headers={'Authorization':'Bearer local-runtime'}) as client:
                    self.assertEqual(client.post('/chat/completions',json={'model':DEEPSEEK}).status_code,200)
                    self.assertEqual(client.post('/rerank',json={'model':QWEN}).status_code,200)
                    self.assertEqual(client.post('/chat/completions',json={'model':QWEN}).status_code,400)
                    self.assertEqual(client.post('/rerank',json={'model':DEEPSEEK}).status_code,400)
                    emb=client.post('/embeddings',json={'model':'BAAI/bge-m3','input':['тест']})
                    self.assertEqual(emb.json()['data'][0]['embedding'],[.1,.2,.3])
                    self.assertEqual(client.post('/embeddings',json={'model':DEEPSEEK,'input':['x']}).status_code,400)
                    self.assertEqual(client.post('/models',json={}).status_code,400)
                    self.assertEqual(client.post('/chat/completions',json={'model':DEEPSEEK},headers={'Authorization':'bad'}).status_code,403)
                self.assertEqual(seen,[('deepseek.invalid','/v1/chat/completions',DEEPSEEK),
                    ('qwen.invalid','/v1/rerank',QWEN)])
            finally:server.shutdown();server.server_close();thread.join();api.close()


if __name__=='__main__':unittest.main()
