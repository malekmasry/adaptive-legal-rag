import unittest
from types import SimpleNamespace
from citation_labels import repair_note_locators,finish_answer,display_citations

class CitationLabelTests(unittest.TestCase):
    def test_readable_labels_require_same_cited_source_hash(self):
        import hashlib
        from unittest.mock import patch
        doc=SimpleNamespace(filename='документы компании__Регламент согласования договоров ред 2023 (утратил силу).txt',raw='Source')
        original='Регламент согласования договоров ред 2023 (утратил силу)'
        answer='Нужна бумага. ('+original+', п. 3.1)'
        ev=[{'source_id':'D1','document':original}]
        book={'documents':[{'filename':doc.filename,'sha256':hashlib.sha256(doc.raw.encode()).hexdigest()}]}
        with patch('corpus_playbook.load_playbook',return_value=book):
            self.assertIn('ред. от 10.01.2023, п. 3.1',display_citations(answer,['D1'],ev,{'d':doc}))
            self.assertEqual(display_citations(answer,[],ev,{'d':doc}),answer)
            doc.raw='Changed'
            self.assertEqual(display_citations(answer,['D1'],ev,{'d':doc}),answer)

    def test_amended_clause_label_does_not_change_number_or_answer(self):
        import hashlib
        from unittest.mock import patch
        original='Доп соглашение 1 к договору П-17-2024'
        doc=SimpleNamespace(filename='документы компании__'+original+'.txt',raw='Source')
        answer='Адрес прежний. ('+original+', п. 3.2, абзац «Новый адрес»)'
        book={'documents':[{'filename':doc.filename,'sha256':hashlib.sha256(doc.raw.encode()).hexdigest()}]}
        with patch('corpus_playbook.load_playbook',return_value=book):
            got=display_citations(answer,['D1'],[{'source_id':'D1','document':original}],{'d':doc})
            self.assertTrue(got.startswith('Адрес прежний.'))
            self.assertIn('новая редакция п. 3.2 договора',got)
            self.assertIn('абзац «Новый адрес»',got)

    def setUp(self):
        self.raw='3. Основная часть.\n10. Штраф.\nПримечания:\n1. Первое примечание.\n3. Категория юридического лица.'
        self.docs={'d':SimpleNamespace(filename='КоАП_ст13.11.txt',raw=self.raw)}
        self.evidence=[{'source_id':'D1','document':'КоАП_ст13.11','text':'10. Штраф.'},
                       {'source_id':'D2','document':'КоАП_ст13.11','text':'3. Категория юридического лица.'}]
        self.answer='Штраф 100–300 тысяч. (КоАП_ст13.11, ч. 10, ч. 3)'
    def test_only_citation_label_changes(self):
        got=repair_note_locators(self.answer,['D1','D2'],self.evidence,self.docs)
        self.assertEqual(got,'Штраф 100–300 тысяч. (КоАП_ст13.11, ч. 10, п. 3 примечаний)')
    def test_unused_note_does_not_rewrite_actual_part(self):
        self.assertEqual(repair_note_locators(self.answer,['D1'],self.evidence,self.docs),self.answer)
    def test_ambiguous_full_source_does_not_get_guessed_correction(self):
        self.evidence[0]['text']=self.raw
        self.assertEqual(repair_note_locators(self.answer,['D1','D2'],self.evidence,self.docs),self.answer)
    def test_changed_or_unseen_quote_does_not_get_correction(self):
        self.docs['d'].raw=self.raw.replace('Категория','Определение')
        self.assertEqual(repair_note_locators(self.answer,['D1','D2'],self.evidence,self.docs),self.answer)
    def test_other_document_and_answer_prose_are_untouched(self):
        answer='По ч. 3 другой нормы. (Иной документ, ч. 3)'
        self.assertEqual(repair_note_locators(answer,['D1','D2'],self.evidence,self.docs),answer)
    def test_decimal_number_is_not_prefix_replaced(self):
        answer='Условие. (КоАП_ст13.11, ч. 3.1)'
        self.assertEqual(repair_note_locators(answer,['D1','D2'],self.evidence,self.docs),answer)

    def test_pipeline_repairs_label_after_final_review(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from pipeline import Pipeline,Answer
        from settings import Settings
        with tempfile.TemporaryDirectory() as temp:
            core=object.__new__(Pipeline);core.s=Settings(cache_dir=Path(temp))
            core.docs=self.docs;core.relations=[];core.all_chunks=[]
            core.decision_index=SimpleNamespace(select=lambda *a,**k:([],{'selected':[],'tokens':0}))
            core.retriever=SimpleNamespace(search=lambda *a,**k:([],{}))
            core.api=SimpleNamespace(structured=lambda *a:Answer(status='answered',paragraphs=[{
                'text':'Штраф 100–300 тысяч.','citations':[{'source_id':'D1','locator':'ч. 10'},{'source_id':'D2','locator':'ч. 3'}]}]))
            with patch('pipeline.context_bundle',return_value=(self.evidence,{'estimated_tokens':100,'budget':18000})):
                got=core.ask('Что установлено?')
            self.assertIn('п. 3 примечаний',got['answer'])

    def test_deadline_only_removes_only_exact_unrequested_ready_draft_section(self):
        draft=('Заявка направляется по электронной почте. В ней указывают наименование, количество товара и желаемую дату поставки. (Договор поставки ПТ-05-2023 Альфа-Трейдинг, п. 1.2)\n\n'
               'Срок поставки — 14 рабочих дней с даты получения заявки поставщиком. (Договор поставки ПТ-05-2023 Альфа-Трейдинг, п. 3.1)')
        short=finish_answer(draft,'Когда поставщик должен привезти товар по ПТ-05?',[],[],{})
        self.assertTrue(short.startswith('Срок поставки'))
        self.assertEqual(finish_answer(draft,'Как направить заявку по ПТ-05 и что указать? Сколько ждать поставку?',[],[],{}),draft)
        modified=draft.replace('14 рабочих','16 рабочих')
        self.assertEqual(finish_answer(modified,'Когда поставщик должен привезти товар по ПТ-05?',[],[],{}),modified)

if __name__=='__main__':unittest.main()
