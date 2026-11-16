"""Add format-stable fingerprints/offsets to the three reviewed source indexes."""
import argparse,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'focused-rag'
sys.path.insert(0,str(ROOT))
from source_identity import fingerprints,canonical_offsets


def build(archive):
    with zipfile.ZipFile(archive) as z:
        sources={Path(n).name:z.read(n).decode('utf-8-sig') for n in z.namelist() if n.endswith('.txt')}
    for filename in ('corpus_playbook.json','decision_catalog.json','relation_scope.json'):
        path=ROOT/filename;data=json.loads(path.read_text(encoding='utf8'))
        if filename=='relation_scope.json':
            for row in data['relations']:
                for side in ('source','target'):
                    exact,normalized=fingerprints(sources[row[side]])
                    assert row[side+'_sha256']==exact
                    row[side+'_canonical_sha256']=normalized
        else:
            for doc in data['documents']:
                exact,normalized=fingerprints(sources[doc['filename']])
                assert doc['sha256']==exact
                doc['canonical_sha256']=normalized
            if filename=='decision_catalog.json':
                docs={d['id']:d['filename'] for d in data['documents']}
                for row in data['units']:
                    row['canonical_start'],row['canonical_end']=canonical_offsets(sources[docs[row['doc_id']]],row['start'],row['end'])
                data['description']='Source offsets, exact and whitespace-normalized fingerprints. Runtime snippets are always literal spans of the supplied live source.'
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return sources


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);build(p.parse_args().archive)
