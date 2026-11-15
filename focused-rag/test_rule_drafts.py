from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from rule_drafts import build_rule_draft,available_sources,number,event_date,FAMILIES
import tempfile
from pathlib import Path
from pipeline import Pipeline,DraftCheck,Answer
from settings import Settings


class RuleDraftTests(unittest.TestCase):
    def draft(self,q,missing=()):
        sources={k:{'source_id':'D'+str(i)} for i,k in enumerate(FAMILIES) if k not in missing}
        with patch('rule_drafts.available_sources',return_value=sources):
            return build_rule_draft(q,[],{},'2026-11-14')

    def text(self,q,**kw):
        d=self.draft(q,**kw)
        return '\n'.join(p['text'] for p in d['paragraphs']) if d else None

    def test_changed_values_and_word_numbers_are_calculated(self):
        self.assertEqual(number('сто тысяч'),Decimal(100000))
        self.assertIn('262,50',self.text('Северному Складу должны 37 500 руб. Просрочка 7 дней. Сколько пени?'))
        self.assertIn('500,00',self.text('За склад Северному Складу должны сто тысяч. Пять дней просрочки. Сколько пени?'))
        self.assertIn('10000,00',self.text('П-17/2024. Неоплаченная сумма 100000 рублей, просрочка оплаты 120 дней. Рассчитать неустойку.'))
        self.assertIn('6000,00',self.text('ПТ-05/2023. Неоплаченная сумма 100000 рублей, просрочка оплаты 120 дней. Рассчитать неустойку.'))

    def test_uncertain_calculation_falls_back(self):
        self.assertIsNone(self.draft('П-17/2024. Долг 100000 рублей, просрочка 120 дней. Сколько пени?'))
        self.assertIsNone(self.draft('Северный Склад: долг 100 рублей, частично погашен, 5 дней просрочки. Сколько пени?'))
        self.assertIsNone(self.draft('Северный Склад: долг 100 рублей или 200 рублей, просрочка 5 дней. Сколько пени?'))

    def test_amendment_boundary_includes_original_day(self):
        a=self.text('Спецификация по П-17/2024 подписана 1 октября 2024 года. Куда доставить партию?')
        b=self.text('Спецификация по П-17/2024 подписана 2 октября 2024 года. Куда доставить партию?')
        self.assertIn('Складская',a);self.assertIn('Заводская',b)
        self.assertIsNone(self.draft('Спецификация по П-17/2024 подписана 30 сентября 2024 года, доставка 15 октября 2024 года. Куда доставить партию?'))

    def test_effective_date_not_letter_date(self):
        a=self.text('Иностранный контрагент, договор подписываем 20 апреля 2025 года. Какие проверки?')
        b=self.text('Иностранный контрагент, договор подписываем 21 апреля 2025 года. Какие проверки?')
        self.assertIn('ещё не обязательна',a);self.assertIn('дополнительно требует',b)
        self.assertEqual(event_date('28 февраля 2025 года','2026-11-14'),date(2025,2,28))
        self.assertIsNone(event_date('31 февраля 2025 года','2026-11-14'))

    def test_two_entities_are_not_merged_and_unknown_sources_do_not_draft(self):
        text=self.text('П-17/2024 и ПТ-05/2023: какая гарантия по каждому?')
        self.assertIn('24 месяца',text);self.assertIn('производитель',text)
        self.assertIsNone(self.draft('П-17/2024: какой срок оплаты?',missing=['p17']))
        doc=SimpleNamespace(filename='документы компании__Договор поставки П-17-2024 Альфа-Трейд.txt',raw='4.2. Оплатить за 17 дней.')
        self.assertEqual(available_sources([],{'d':doc}),{})

    def test_all_paragraphs_have_known_provision_citations(self):
        draft=self.draft('Ноябрь 2026-го. Налоговый штраф, одно смягчающее и отягчающее обстоятельство по статье 112. Сравнить.')
        self.assertEqual(draft['status'],'answered')
        for p in draft['paragraphs']:
            self.assertTrue(p['citations']);self.assertTrue(all(r['locator'] for r in p['citations']))

    def test_no_ready_variant_generates_then_reviews_the_generated_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            core=object.__new__(Pipeline);core.s=Settings(cache_dir=Path(tmp))
            core.docs={'d':SimpleNamespace(id='d',filename='test.txt',raw='1. Срок 45 дней от поставки.',title='test')};core.relations=[]
            core.retriever=SimpleNamespace(search=lambda *a,**k:([SimpleNamespace(doc_id='d')],{}))
            core.all_chunks=[];core.decision_index=SimpleNamespace(select=lambda *a,**k:([],{'selected':[],'tokens':0}))
            calls=[]
            def model(prompt,payload,schema):
                calls.append((schema,payload))
                text='Срок 45 дней.' if len(calls)==1 else 'Срок 45 дней от поставки.'
                return Answer(status='answered',paragraphs=[{'text':text,'citations':[{'source_id':'D1','locator':'п. 1'}]}])
            core.api=SimpleNamespace(structured=model)
            with patch('pipeline.build_rule_draft',side_effect=AssertionError('Ready draft must not run')):
                result=core.ask('Какой срок?')
            self.assertEqual([s for s,p in calls],[Answer,Answer])
            self.assertEqual(calls[1][1]['previous_draft']['paragraphs'][0]['text'],'Срок 45 дней.')
            self.assertEqual(result['trace']['draft_origin'],'generator')
            self.assertIn('от поставки',result['answer'])


if __name__=='__main__':unittest.main()
