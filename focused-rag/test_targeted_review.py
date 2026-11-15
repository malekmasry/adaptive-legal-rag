import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pydantic import ValidationError
from targeted_review import PatchReview,apply_patches
from pipeline import Answer,Paragraph,Reference,Pipeline
from question_brief import answer_schema
from settings import Settings


class TargetedReviewTests(unittest.TestCase):
    evidence=[{'source_id':'D1','document':'Договор','text':'1. Срок 45 дней от поставки.\n2. Сохраняется исключение для возврата.'}]
    def draft(self,text='Срок 40 дней. Правильное условие сохраняется.'):
        return Answer(status='answered',paragraphs=[{'text':text,'citations':[{'source_id':'D1','locator':'п. 1'}]}])
    def edit(self,**kw):
        return {'paragraph':0,'before':'40','after':'45','reason':'factual_error',
                'evidence':[{'source_id':'D1','quote':'Срок 45 дней от поставки.'}],**kw}
    def test_empty_patch_list_preserves_object_and_exact_text(self):
        draft=self.draft();result,report=apply_patches(draft,self.evidence,PatchReview(patches=[]))
        self.assertIs(result,draft);self.assertTrue(report['draft_preserved'])
    def test_only_target_changes_with_source_exact_locator(self):
        draft=self.draft();before=draft.model_dump()
        revised,report=apply_patches(draft,self.evidence,PatchReview(patches=[self.edit()]))
        self.assertEqual(revised.paragraphs[0].text,'Срок 45 дней. Правильное условие сохраняется.')
        self.assertEqual(draft.model_dump(),before);self.assertEqual(report['applied'],1)
        self.assertEqual(revised.paragraphs[0].citations[-1].locator,'абзац «1. Срок 45 дней от поставки.»')
    def test_known_catalog_locator_is_reused(self):
        catalog=[{'source_id':'D1','citation_id':'C1','locator':'п. 1','quote':'1. Срок 45 дней от поставки.'}]
        revised,_=apply_patches(self.draft(),self.evidence,PatchReview(patches=[self.edit()]),catalog)
        self.assertEqual(len(revised.paragraphs[0].citations),1)
    def test_bad_quote_unknown_source_and_unsupported_replacement_reject(self):
        for evidence in ([],[{'source_id':'D9','quote':'Срок 45 дней от поставки.'}],
                         [{'source_id':'D1','quote':'Срок 99 дней от поставки.'}]):
            draft=self.draft();saved=draft.model_dump()
            with self.assertRaises(ValueError):apply_patches(draft,self.evidence,PatchReview(patches=[self.edit(evidence=evidence)]))
            self.assertEqual(saved,draft.model_dump())
    def test_ambiguous_or_overlapping_target_is_rejected_atomically(self):
        with self.assertRaises(ValueError):
            apply_patches(self.draft('40 и 40'),self.evidence,PatchReview(patches=[self.edit()]))
        draft=self.draft();saved=draft.model_dump()
        with self.assertRaises(ValueError):
            apply_patches(draft,self.evidence,PatchReview(patches=[self.edit(),self.edit(before='Срок 40',after='Срок 45')]))
        self.assertEqual(saved,draft.model_dump())
    def test_irrelevant_deletion_preserves_remaining_characters(self):
        draft=self.draft('Срок 45 дней. Посторонний совет.')
        revised,_=apply_patches(draft,self.evidence,PatchReview(patches=[self.edit(before='Посторонний совет.',after='',reason='irrelevant',evidence=[])]))
        self.assertEqual(revised.paragraphs[0].text,'Срок 45 дней. ')
    def test_deletion_cannot_erase_entire_answer_or_requested_part(self):
        patch=self.edit(before='Срок 40 дней.',after='',reason='irrelevant',evidence=[])
        with self.assertRaises(ValueError):apply_patches(self.draft('Срок 40 дней.'),self.evidence,PatchReview(patches=[patch]))
        schema=answer_schema(Answer,Paragraph,Reference,{'parts':[{'part_id':'P1'},{'part_id':'P2'}]},[{'citation_id':'C1'}])
        original=schema(status='answered',paragraphs=[{'part_id':p,'text':'Срок 40 дней.','citations':[{'source_id':'D1','locator':'п. 1'}]} for p in ('P1','P2')])
        with self.assertRaises(ValueError):apply_patches(original,self.evidence,PatchReview(patches=[patch]))
    def test_invalid_deletion_reason_and_missing_schema_field_fail(self):
        with self.assertRaises(ValueError):apply_patches(self.draft(),self.evidence,PatchReview(patches=[self.edit(after='',reason='missing_condition')]))
        with self.assertRaises(ValidationError):PatchReview.model_validate({})
    def test_live_changed_source_is_authoritative_for_quote(self):
        changed=copy.deepcopy(self.evidence);changed[0]['text']=changed[0]['text'].replace('45','52')
        with self.assertRaises(ValueError):apply_patches(self.draft(),changed,PatchReview(patches=[self.edit()]))
    @patch('pipeline.prepare_question_brief',return_value=None)
    def test_pipeline_rejected_patch_keeps_valid_generated_answer(self,_brief):
        with tempfile.TemporaryDirectory() as tmp:
            core=object.__new__(Pipeline);core.s=Settings(cache_dir=Path(tmp));core.relations=[];core.all_chunks=[]
            core.docs={'d':SimpleNamespace(id='d',filename='Договор.txt',raw=self.evidence[0]['text'],title='Договор')}
            core.retriever=SimpleNamespace(search=lambda *a,**k:([SimpleNamespace(doc_id='d')],{}))
            core.decision_index=SimpleNamespace(select=lambda *a,**k:([],{'selected':[],'tokens':0}))
            calls=[]
            def model(instruction,payload,schema):
                calls.append(schema)
                if schema is PatchReview:return PatchReview(patches=[self.edit(before='999',after='1000')])
                return self.draft('Срок 45 дней от поставки.')
            core.api=SimpleNamespace(structured=model)
            with patch('pipeline.build_rule_draft',return_value=None):result=core.ask('Каков срок по договору?')
            self.assertEqual(len(calls),2);self.assertEqual(result['status'],'answered')
            self.assertEqual(result['trace']['review'],'failed_kept_draft');self.assertIn('45 дней',result['answer'])


if __name__=='__main__':unittest.main()
