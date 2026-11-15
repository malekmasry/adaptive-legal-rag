"""Compile a source-bound location index; no benchmark inputs or source bodies."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'focused-rag'
sys.path.insert(0,str(ROOT))
from corpus import ingest,digest
from answer_guides import plan_for

def build(data):
    docs,chunks,_=ingest(data);by_id={c.id:c for c in chunks};rows=[]
    for c in chunks:
        if c.kind!='passage' or c.locator.startswith('раздел '):continue
        parent=by_id.get(c.parent_id)
        scope=parent if parent and parent.kind=='parent' else c
        rows.append({'id':c.id,'doc_id':c.doc_id,'start':scope.start,'end':scope.end,
            'search_start':c.start,'search_end':c.end,'locator':c.locator,
            'topics':plan_for(c.text)['topics']})
    result={'version':1,'documents':[{'id':d.id,'filename':d.filename,'sha256':digest(d.raw)} for d in docs],
            'units':rows,'description':'Source offsets only. A unit is used only against an exact live source hash; full governing scopes are retained.'}
    (ROOT/'decision_catalog.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'documents':len(docs),'indexed_units':len(rows),'source_bytes_in_index':0}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);build(p.parse_args().data)
