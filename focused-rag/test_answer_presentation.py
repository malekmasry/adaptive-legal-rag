import unittest
from types import SimpleNamespace
from unittest.mock import patch
from answer_presentation import render_answer,chars,instruction_for_answer,PRESENTATION,MAX_CHARS
from pipeline import Answer,INSTRUCTION


class PresentationTests(unittest.TestCase):
    def evidence(self):
        return [{'source_id':'D1','document':'Договор',
                 'text':'1. При подтверждённом условии срок составляет 45 календарных дней от поставки.'},
                {'source_id':'D2','document':'Другой договор','text':'2. Срок составляет 90 дней.'}]

    def draft(self,**changes):
        paragraph={'text':'При подтверждённом условии — 45 календарных дней от поставки.',
            'explanation':'Условие задано выполненным. Поэтому применяется срок от поставки, а дата счёта не меняет предусмотренное начальное событие.',
            'source_quote':'срок составляет 45 календарных дней от поставки.',
            'citations':[{'source_id':'D1','locator':'п. 1'}]}
        paragraph.update(changes)
        return Answer(status='answered',paragraphs=[paragraph])

    def render(self,draft=None,mode='detailed',**kwargs):
        return render_answer(draft or self.draft(),[],self.evidence(),{},'Когда платить?','2026-11-16',mode,**kwargs)

    def test_opposite_views_preserve_identical_complete_core_and_sources(self):
        draft=self.draft();a,ra,ma=self.render(draft,'concise');b,rb,mb=self.render(draft,'detailed')
        self.assertEqual(ra,rb)
        for text in [a,b]:
            self.assertIn(draft.paragraphs[0].text,text)
            self.assertIn('(Договор, п. 1)',text)
            self.assertTrue(text.endswith('По состоянию на текущую дату.'))
        self.assertNotIn('Обоснование:',a);self.assertNotIn('Из источника:',a)
        self.assertIn(draft.paragraphs[0].explanation,b);self.assertIn(draft.paragraphs[0].source_quote,b)
        self.assertEqual(ma['core_chars'],mb['core_chars']);self.assertGreater(mb['chars'],ma['chars'])

    def test_quote_must_be_in_a_cited_live_source(self):
        for quote in ['Неверная придуманная цитата.','2. Срок составляет 90 дней.','срок составляет 46 календарных дней']:
            text,_,m=self.render(self.draft(source_quote=quote))
            self.assertNotIn(quote,text);self.assertEqual(m['invalid_quotes_omitted'],1)
            self.assertIn('45 календарных дней',text)

    def test_whitespace_quote_is_rendered_as_the_exact_live_span(self):
        quote='срок\tсоставляет  45 календарных дней от поставки.'
        text,_,m=self.render(self.draft(source_quote=quote))
        self.assertIn('«срок составляет 45 календарных дней от поставки.»',text)
        self.assertEqual(m['quotes'],1)

    def test_all_core_parts_survive_overlong_explanations_without_slicing(self):
        paragraphs=[]
        for i in range(8):
            paragraphs.append({'text':f'Вывод {i}: '+('существенное условие '*22),
                'explanation':f'Пояснение {i}: '+('Детальное обоснование. '*90)+'КОНЕЦ_'+str(i),
                'source_quote':'','citations':[{'source_id':'D1','locator':'п. 1'}]})
        draft=Answer(status='answered',paragraphs=paragraphs)
        text,refs,m=self.render(draft)
        self.assertLessEqual(chars(text),MAX_CHARS)
        self.assertGreater(m['explanations_omitted_for_length'],0)
        self.assertEqual(refs,['D1'])
        for p in paragraphs:
            self.assertIn(p['text'].strip(),text)
            if p['explanation'][:15] in text:self.assertIn(p['explanation'],text)
        self.assertEqual(text.count('(Договор, п. 1)'),8)

    def test_footer_and_citations_are_reserved_before_optional_detail(self):
        core,_,_=self.render(mode='concise')
        text,_,m=self.render(max_chars=chars(core))
        self.assertEqual(text,core);self.assertEqual(m['explanations'],0);self.assertEqual(m['quotes'],0)

    def test_utf16_count_also_protects_non_bmp_character_budget(self):
        draft=self.draft(explanation='😀'*1000,source_quote='')
        core,_,_=self.render(draft,'concise')
        text,_,m=self.render(draft,max_chars=chars(core)+1200)
        self.assertNotIn('😀',text);self.assertEqual(m['explanations_omitted_for_length'],1)
        self.assertEqual(chars('😀'),2)

    def test_impossible_core_is_rejected_without_deleting_required_facts(self):
        draft=self.draft();saved=draft.model_dump()
        with self.assertRaisesRegex(ValueError,'Complete core'):self.render(draft,max_chars=50)
        self.assertEqual(draft.model_dump(),saved)

    def test_historical_scope_is_not_replaced_by_current_date(self):
        text,_,_=render_answer(self.draft(),[],self.evidence(),{},'Когда платить на 15.03.2025?','2026-11-16','detailed')
        self.assertNotIn('текущую дату',text)

    def test_partial_gap_is_kept_in_both_views(self):
        draft=Answer(status='partial',paragraphs=[self.draft().paragraphs[0].model_dump(),
            {'text':'Срок уничтожения по отсутствующей статье установить нельзя.','citations':[],
             'explanation':'Нужный текст статьи в предоставленных источниках отсутствует.','source_quote':''}])
        for mode in ['concise','detailed']:
            text,_,_=self.render(draft,mode)
            self.assertIn('45 календарных дней',text);self.assertIn('установить нельзя',text)

    def test_generator_contract_does_not_depend_on_rendering_mode(self):
        with patch('answer_presentation.output_mode',return_value='concise'):a=instruction_for_answer(INSTRUCTION)
        with patch('answer_presentation.output_mode',return_value='detailed'):b=instruction_for_answer(INSTRUCTION)
        self.assertEqual(a,b);self.assertIn(PRESENTATION,a)
        self.assertNotIn('до 150 слов суммарно',a)

    def test_catalog_references_resolve_in_both_views(self):
        paragraph={'part_id':'P1','text':'Срок — 45 дней.','explanation':'Начальное событие установлено договором.',
                   'source_quote':'срок составляет 45 календарных дней от поставки.',
                   'citations':[{'citation_id':'C1'}]}
        draft=SimpleNamespace(paragraphs=[SimpleNamespace(model_dump=lambda:paragraph)])
        catalog=[{'citation_id':'C1','source_id':'D1','locator':'п. 1'}]
        for mode in ['concise','detailed']:
            text,refs,_=render_answer(draft,catalog,self.evidence(),{},'Срок?','2026-11-16',mode)
            self.assertEqual(refs,['D1']);self.assertIn('(Договор, п. 1)',text);self.assertNotIn('C1',text)


if __name__=='__main__':unittest.main()
