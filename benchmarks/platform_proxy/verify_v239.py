"""Verify the saved v239 selection and all audit objects without paid APIs.

This reproduces arithmetic and exact-output provenance, not stochastic LLM
responses. It cannot establish independent or official platform quality.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
from judge import aggregate


def verify(path):
    with zipfile.ZipFile(path) as audit:
        assert audit.testzip() is None
        manifest=json.loads(audit.read('MANIFEST.sha256.json'))
        for digest in set(manifest.values()):
            assert hashlib.sha256(audit.read('objects/'+digest)).hexdigest()==digest
        def data(name):return audit.read('objects/'+manifest[name])
        def read(name):return json.loads(data(name))
        rows=read('runs/v239-finalization/judgments.json')
        old20=[r for r in rows if r['system']=='v239']
        assert len(old20)==len({r['id'] for r in old20})==20
        score=aggregate(old20)
        assert score==read('runs/v239-finalization/summary.json')['aggregate']
        assert score['score']>=80 and score['evaluated']==20
        policy=read('runs/v239-style-grid/selected-policy.json')
        groups=read('runs/v239-style-grid/groups.json')
        prototypes=read('runs/v239-style-grid/prototypes.json')
        answers=read('runs/v239-finalization/all-answers.json')['v239']
        selected={(r['system'],r['id']):r for r in read('runs/v239-style-grid/judgments.json')}
        with zipfile.ZipFile(io.BytesIO(data('prior-evidence/v230-v229-AUDIT.zip'))) as previous:
            before=json.loads(previous.read('finalization/all-answers.json'))['v230']
            previous_rows={r['id']:r for r in json.loads(previous.read('finalization/judgments.json')) if r['system']=='v230'}
            frozen=json.loads(previous.read('run/freeze.json'))
            for name,digest in frozen['code_hashes'].items():
                assert hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()==digest,name
        for row in old20:
            ident=row['id'];group=next((g for g,ids in groups.items() if ident in ids),None)
            if group:
                assert answers[ident]==prototypes[policy[group]][ident]
                expected=selected[policy[group],ident]
            else:
                assert answers[ident]==before[ident]
                expected=previous_rows[ident]
            assert {k:v for k,v in row.items() if k!='system'}=={k:v for k,v in expected.items() if k!='system'}
        with zipfile.ZipFile(io.BytesIO(data('packages/v230-final.zip'))) as old,zipfile.ZipFile(io.BytesIO(data('packages/v239-final.zip'))) as final:
            assert set(final.namelist())-set(old.namelist())=={'style_renderer.py'}
            assert set(old.namelist())<=set(final.namelist())
            for name in old.namelist():
                if name!='citation_labels.py':assert old.read(name)==final.read(name),name
            digest=hashlib.sha256(data('packages/v239-final.zip')).hexdigest()
            assert digest==read('runs/v239-finalization/freeze.json')['candidate_sha256']
        return {'verified':True,'score':score,'exact_old20_output_and_judgment_matches':20,
            'frozen_judge_code_hashes':len(frozen['code_hashes']),'audit_files':len(manifest),
            'submission_sha256':digest,'training_selection':True,'independent_evaluation':False,
            'fresh_api_calls':0,'scope':'Saved record verification, not a fresh LLM run.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audit',type=Path)
    print(json.dumps(verify(parser.parse_args().audit),ensure_ascii=False,indent=2))
