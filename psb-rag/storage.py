import json, threading, time
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from dataclasses import asdict

class Store:
    def __init__(self,path):
        self.engine=create_engine('sqlite:///'+str(path),connect_args={'check_same_thread':False,'timeout':30})
        self._locks={}; self._guard=threading.Lock()
        with self.engine.begin() as c:
            c.exec_driver_sql('PRAGMA journal_mode=WAL')
            for table in ('documents','chunks','relations','sessions'):
                c.exec_driver_sql(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    def close(self): self.engine.dispose()
    def put(self,table,id,payload):
        assert table in {'documents','chunks','relations','sessions'}
        with self.engine.begin() as c:
            c.execute(text(f'INSERT INTO {table} VALUES (:id,:p) ON CONFLICT(id) DO UPDATE SET payload=:p'),{'id':id,'p':json.dumps(payload,ensure_ascii=False)})
    def sync_corpus(self,docs,chunks,relations):
        with self.engine.begin() as c:
            for table,items in [('documents',[asdict(d) for d in docs]),('chunks',[asdict(x) for x in chunks]),('relations',relations)]:
                c.execute(text(f'DELETE FROM {table}'))
                for i,item in enumerate(items):
                    c.execute(text(f'INSERT INTO {table} VALUES (:id,:p)'),{'id':item.get('id',str(i)),'p':json.dumps(item,ensure_ascii=False)})
    @contextmanager
    def session_lock(self,id):
        with self._guard: lock=self._locks.setdefault(id,threading.Lock())
        with lock: yield
    def session(self,id,ttl):
        with self.engine.connect() as c:
            row=c.execute(text('SELECT payload FROM sessions WHERE id=:id'),{'id':id}).first()
        if row:
            s=json.loads(row[0])
            if time.time()-s['updated']<ttl: return s
        return {'turns':[],'summary':'','slots':{},'updated':time.time()}
    def save_session(self,id,s,question,answer,slots,summary=None):
        turns=s['turns']+[{'user':question,'assistant':answer}]
        # Summary consists solely of old user questions; never treated as evidence.
        if len(turns)>3:
            s['summary']=summary if summary is not None else (s.get('summary','')+'\n'+turns[0]['user'])[-1800:]
        s.update(turns=turns[-3:],slots=slots,updated=time.time())
        self.put('sessions',id,s)
    def purge_sessions(self,ttl):
        now=time.time()
        with self.engine.begin() as c:
            rows=c.execute(text('SELECT id,payload FROM sessions')).all()
            for id,p in rows:
                if now-json.loads(p)['updated']>=ttl: c.execute(text('DELETE FROM sessions WHERE id=:id'),{'id':id})
