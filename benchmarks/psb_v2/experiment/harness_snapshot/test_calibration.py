import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zipfile import ZipFile

from benchmark import cases
from calibrate import analyze, ranks, score
from judge import combine, validate_proofs, validate_judgment
from run_matrix import ROOT, export_questions, freeze
from live_run import merge_interrupted_vector_run


class CalibrationTests(unittest.TestCase):
    def test_equal_answers_are_fifty_not_one_hundred(self):
        self.assertEqual(score([.5, .5, .5]), 50)
        self.assertEqual(score([0, 0]), 0)
        self.assertEqual(ranks([3, 1, 3]), [2.5, 1, 2.5])

    def test_missing_judge_results_cannot_silently_improve_ranking_gate(self):
        dataset = [{'id':str(i), 'family_id':str(i//2)} for i in range(10)]
        judgments = {v:[{'id':str(i), 'status':'ok', 'score':value} for i in range(10)]
                     for v,value in [('a',.125),('b',.25),('c',.625),('d',.875)]}
        judgments['d'][0] = {'id':'0','status':'needs_review'}
        report = analyze(dataset,judgments,{'a':14,'b':22.57,'c':29,'d':35},50)
        self.assertEqual(report['common_coverage'],.9)
        self.assertFalse(report['historical_order_reproduced'])
        self.assertEqual(report['coverage']['d']['unknown'],1)
        self.assertLess(*report['coverage']['d']['all_case_score_bounds'])

    def test_inverted_order_and_same_scores_are_detected(self):
        dataset = [{'id':'x','family_id':'f'}, {'id':'y','family_id':'g'}]
        for values in ([1,.75,.25,0],[.5,.5,.5,.5]):
            judgments={v:[{'id':r['id'],'status':'ok','score':value} for r in dataset]
                       for v,value in zip('abcd',values)}
            report=analyze(dataset,judgments,dict(zip('abcd',[14,22.57,29,35])),50)
            self.assertFalse(report['historical_order_reproduced'])
            self.assertEqual(report['spearman'],-1 if values[0]==1 else None)

    def test_duplicate_and_invalid_scores_fail(self):
        dataset=[{'id':'x','family_id':'f'}]
        for rows in ([{'id':'x','status':'ok','score':.9}],
                     [{'id':'x','status':'ok','score':.5}]*2):
            with self.assertRaises(ValueError):analyze(dataset,{'a':rows},{'a':14},10)

    def test_blinded_questions_export_has_no_gold_or_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'questions.csv'
            ids=export_questions(path,'development')
            with path.open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(set(rows[0]),{'id','question'})
            self.assertEqual(len(ids),63)
            self.assertTrue(set(ids).isdisjoint(row['id'] for row in cases('holdout')))
            with self.assertRaises(ValueError):export_questions(path,'development',['S03a'])

    def test_variants_never_leak_across_family_splits(self):
        families={}
        for row in cases():families.setdefault(row['family_id'],set()).add(row['split'])
        self.assertTrue(all(len(splits)==1 for splits in families.values() if 'holdout' in splits))
        by_id={row['id']:row for row in cases()}
        hard=[row for row in by_id.values() if row['origin']=='v2_hard_paraphrase']
        self.assertEqual(len(hard),21)
        for row in hard:
            parent=by_id[row['parent_case_id']]
            self.assertEqual(row['evidence'],parent['evidence'])
            self.assertEqual(row['family_id'],parent['family_id'])
            self.assertEqual(row['split'],parent['split'])

    def test_judge_cannot_fabricate_its_evidence(self):
        answer='Срок составляет 90 дней.'
        proof={'verdict':'confirmed','fabricated_claim':answer,
               'evidence':[{'path':'contract','quote':'Срок составляет 30 календарных дней.'}]}
        report={'A':{'verdict':'none'},'B':proof}
        payload={'A':'','B':answer,'source_documents':{'contract':'Срок составляет 30 календарных дней.'}}
        validate_proofs(report,payload)
        proof['evidence'][0]['quote']='Срок составляет 90 календарных дней.'
        with self.assertRaises(ValueError):validate_proofs(report,payload)

    def test_literal_quotes_accept_wrappers_but_not_rewording(self):
        answer='Срок составляет 90 дней.'
        proof={'verdict':'confirmed','fabricated_claim':'«'+answer+'»',
               'evidence':[{'path':'contract','quote':'"Срок составляет 30 календарных дней."'}]}
        report={'A':{'verdict':'none'},'B':proof}
        payload={'A':'','B':answer,'source_documents':{'contract':'Срок составляет 30 календарных дней.'}}
        validate_proofs(report,payload)
        proof['fabricated_claim']='«Срок равен 90 дням.»'
        with self.assertRaises(ValueError):validate_proofs(report,payload)

    def test_judge_score_cannot_favor_only_confirmed_fabrication(self):
        clean={'verdict':'none','fabricated_claim':'','evidence':[],
               'missing_facts':[],'citation_errors':[]}
        fabricated={**clean,'verdict':'confirmed','fabricated_claim':'90 дней',
                    'evidence':[{'path':'contract','quote':'30 дней'}]}
        result={'A':clean,'B':fabricated,'score_b':0,'dataset_issue':False,'reason':'Ошибка B.'}
        validate_judgment(result)
        result['score_b']=1
        with self.assertRaises(ValueError):validate_judgment(result)
        result.update(A=fabricated,B=clean,score_b=0)
        with self.assertRaises(ValueError):validate_judgment(result)

    def test_unverified_baseline_cannot_make_candidate_a_verified_winner(self):
        first={'dataset_issue':False,'A':{'verdict':'unverified'},'B':{'verdict':'none'},'score_b':1}
        second={'dataset_issue':False,'A':{'verdict':'none'},'B':{'verdict':'unverified'},'score_b':0}
        self.assertEqual(combine(first,second)['status'],'needs_review')

    def test_resume_excludes_unfinished_placeholders_and_disclaims_batch_timing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output=root/'vector14';old=root/'segments'/'old';new=root/'segments'/'new'
            for path in (output,old,new):path.mkdir(parents=True)
            meta={'version_id':'vector14','commit':'commit','package_sha256':'package',
                  'dataset_sha256':'dataset','corpus_sha256':'corpus','service_profile_sha256':'services','ids':['x','y']}
            (old/'run.meta.json').write_text(json.dumps(meta))
            (new/'run.meta.json').write_text(json.dumps({**meta,'ids':['y'],'exit_code':0,'service_check_passed':True}))
            (old/'submission.csv').write_text('id,answer\nx,completed\ny,unfinished placeholder\n')
            (new/'submission.csv').write_text('id,answer\ny,recovered\n')
            (old/'submission.metrics.jsonl').write_text(json.dumps({'id':'x','status':'answered'})+'\n')
            (new/'submission.metrics.jsonl').write_text(json.dumps({'id':'y','status':'answered'})+'\n')
            merge_interrupted_vector_run(output,old,new)
            with (output/'submission.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows,[{'id':'x','answer':'completed'},{'id':'y','answer':'recovered'}])
            recovered=json.loads((output/'run.meta.json').read_text())
            self.assertTrue(recovered['interrupted_and_resumed'])
            self.assertFalse(recovered['single_batch_timing_valid'])
            self.assertIsNone(recovered['elapsed_seconds'])

    def test_actual_historical_entry_points_and_enabled_review(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path,request))
                if self.path.endswith('/embeddings'):
                    value={'data':[{'index':i,'embedding':[1,0]} for i in range(len(request['input']))]}
                elif self.path.endswith('/rerank'):
                    value={'results':[{'index':i,'relevance_score':.9} for i in range(len(request['documents']))]}
                else:
                    payload=json.loads(request['messages'][1]['content'])
                    content=json.dumps({'verdict':'keep'}) if 'draft_answer' in payload else 'Договор заключён на неопределённый срок [S0001].'
                    value={'choices':[{'finish_reason':'stop','message':{'content':content}}]}
                body=json.dumps(value).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen=root/'frozen'
            if (ROOT/'frozen'/'manifest.json').exists():
                shutil.copytree(ROOT/'frozen',frozen)
                versions=json.loads((frozen/'manifest.json').read_text())['versions']
            else:
                versions=freeze(ROOT.parents[1],frozen)
            self.assertEqual(len(versions),4)
            for item in versions:
                with ZipFile(frozen/item['package']) as z:
                    self.assertNotIn('cases.jsonl',z.namelist())
                    self.assertNotIn('judge.py',z.namelist())
            cfg=root/'services.json'
            cfg.write_text(json.dumps({'LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1',
                                      'LLM_API_KEY':'fixture-secret-key','LLM_MODEL':'stub','EMB_MODEL':'stub',
                                      'RERANKER_MODEL':'stub','HTTP_TRUST_ENV':'0'}))
            for version in ('vector14','review29'):
                calls.clear()
                process=subprocess.run([sys.executable,str(ROOT/'run_matrix.py'),'run','--frozen',str(frozen),
                                        '--version',version,'--services',str(cfg),'--ids','S01a',
                                        '--output',str(root/version)],capture_output=True,text=True,timeout=30,
                                       env={**os.environ,'ANSWER_REVIEW_ENABLED':'0','CONDITIONAL_DENSE_MERGE':'0','MAX_TOKENS':'123'})
                self.assertEqual(process.returncode,0,process.stderr)
                meta=json.loads((root/version/'run.meta.json').read_text())
                self.assertTrue(meta['service_check_passed'])
                self.assertNotIn('fixture-secret-key',json.dumps(meta))
                chats=[request for path,request in calls if path.endswith('/chat/completions')]
                self.assertEqual(len(chats),1 if version=='vector14' else 2)
                self.assertEqual(chats[0]['max_tokens'],1600)
                self.assertFalse((root/version/'cases.jsonl').exists())


if __name__=='__main__':unittest.main()
