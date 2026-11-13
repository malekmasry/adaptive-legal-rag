import csv,json,os,subprocess,sys,tempfile,threading,unittest,zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from build_submission import build,ROOT

class Handler(BaseHTTPRequestHandler):
    routes=[]
    requests=[]
    def log_message(self,*args):pass
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.routes.append(self.path)
        self.requests.append((self.path,data))
        if self.path.endswith('/embeddings'):
            result={'data':[{'index':i,'embedding':[1.0,0.5,0.25]} for i,_ in enumerate(data['input'])]}
            return self.respond(result)
        if self.path.endswith('/rerank'):
            ranked=sorted(range(len(data['documents'])),key=lambda i:'450 000' not in data['documents'][i])[:data['top_n']]
            return self.respond({'results':[{'index':i,'relevance_score':1/(n+1)} for n,i in enumerate(ranked)]})
        content=data['messages'][0]['content'];payload=json.loads(data['messages'][1]['content'])
        if 'Восстанови самостоятельный' in content:
            result={'question':payload['question']}
        elif 'Проверь каждый claim' in content:
            result={'supported_indices':[0],'issues':[],'coverage':[True]}
        else:
            source=next(x for x in payload['evidence'] if '450 000' in x['text'])
            result={'status':'answered','claims':[{'text':'Арендная плата составляет 450000 рублей в месяц','citations':[{'chunk_id':source['chunk_id'],'quote':'450 000 (четыреста пятьдесят тысяч) рублей в месяц, включая НДС.'}]}]}
        self.respond({'choices':[{'message':{'content':json.dumps(result,ensure_ascii=False)},'finish_reason':'stop'}]})
    def respond(self,result):
        body=json.dumps(result).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

class TestPackage(unittest.TestCase):
    def test_R46_R53_R54_execute_extracted_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);archive=build(work/'submit.zip')
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip());self.assertTrue(all('/' not in x for x in z.namelist()));z.extractall(work/'extracted')
            data=work/'input';data.mkdir()
            (data/'notes.txt').write_text('This is not the knowledge base.',encoding='utf8')
            with zipfile.ZipFile(data/'knowledge-base.zip','w') as z:
                for f in (ROOT/'data'/'knowledge-base').glob('*.txt'):z.write(f,f.name)
            with (data/'questions.csv').open('w',encoding='utf8',newline='') as f:
                writer=csv.writer(f);writer.writerow(['id','question']);writer.writerow(['q1','Арендная плата по А-08/2024?']);writer.writerow(['q2','Арендная плата по А-08/2024?'])
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            env={**os.environ,'DATA_DIR':str(data),'CACHE_DIR':str(work/'cache'),'LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}',
                 'LLM_API_KEY':'test-secret','LLM_MODEL':'contract-stub','EMB_MODEL':'contract-stub-vector','RERANKER_MODEL':'platform-reranker',
                 'RETRIEVAL_MODE':'hybrid','USE_RERANK':'1','PYTHONIOENCODING':'utf-8','CHUNKER':'common',
                 'TOP_K':'40','RERANKER_TOP_N':'8','MAX_TOKENS':'1024','MAX_WORKERS':'2'}
            env.pop('PYTHONPATH',None)
            for name in ('RAG_CHUNKER','RAG_TOP_K','RAG_RERANK_TOP_N','EMBEDDER_BASE_URL','RERANKER_BASE_URL','EMBEDDER_API_KEY','RERANKER_API_KEY'):
                env.pop(name,None)
            profiles=[{},
                      {'LLM_BASE_URL':'LLM_BASE_URL','OPENAI_BASE_URL':env['LLM_BASE_URL']+'/v1/',
                       'LLM_API_KEY':'LLM_API_KEY','OPENAI_API_KEY':'test-secret',
                       'LLM_MODEL':'LLM_MODEL','GEN_MODEL':' ','OPENAI_MODEL':'contract-stub'},
                      {'MAX_WORKERS':'MAX_WORKERS','MAX_TOKENS':'invalid','TEMPERATURE':'nan','TOP_K':'TOP_K','RERANKER_TOP_N':'RERANKER_TOP_N',
                       'RAG_CHUNKER':'obsolete-unsupported-value'}]
            try:
                for i,profile in enumerate(profiles):
                    with self.subTest(profile=i):
                        Handler.routes=[];Handler.requests=[]
                        cwd=work/'extracted' if i==0 else work/f'cwd-{i}'
                        cwd.mkdir(exist_ok=True)
                        runtime={**env,**profile,'CACHE_DIR':str(work/f'cache-{i}')}
                        # Cold imports and TLS trust-store loading on Windows can
                        # exceed 100s; this is the test harness limit, not the run budget.
                        proc=subprocess.run([sys.executable,str(work/'extracted'/'solution.py')],cwd=cwd,env=runtime,capture_output=True,text=True,timeout=300)
                        self.assertEqual(proc.returncode,0,proc.stderr)
                        with (cwd/'submission.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
                        self.assertEqual({r['id'] for r in rows},{'q1','q2'});self.assertTrue(all('450000' in r['answer'] for r in rows))
                        self.assertTrue(all('п. 3.1' in r['answer'] for r in rows))
                        self.assertTrue(all(any(route.endswith(suffix) for route in Handler.routes) for suffix in ('/chat/completions','/embeddings','/rerank')))
                        for route,payload in Handler.requests:
                            if route.endswith('/rerank'):
                                self.assertEqual(payload['top_n'],len(payload['documents']))
                                self.assertEqual(payload['top_n'],64)
                                self.assertNotIn('<Instruct>',payload['query'])
                                self.assertEqual(payload['model'],'platform-reranker')
                            elif route.endswith('/chat/completions'):
                                self.assertEqual(payload['model'],'contract-stub')
                                self.assertEqual(payload['max_tokens'],8192 if i==2 else 1024)
            finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main(verbosity=2)
