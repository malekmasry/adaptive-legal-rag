import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from relation_scope import expand_relations
from focused_context import context_bundle,ScopeRetriever


class RelationScopeTests(unittest.TestCase):
    def setUp(self):
        def doc(i,raw):return SimpleNamespace(id=i,filename=i+'.txt',raw=raw,title=i)
        self.docs={'base':doc('base','3.2. Старый адрес.\n5.2. Гарантия 24 месяца.'),
                   'change':doc('change','Изменить пункт 3.2.\nНовый адрес применяется после даты спецификации.\nДругие условия без изменений.')}
        self.rel={'source':'change','target':'base','kind':'amends'}
        self.rule={**self.rel,'source':'change.txt','target':'base.txt','topics':['delivery_address','amendment_question'],
          'source_sha256':hashlib.sha256(self.docs['change'].raw.encode()).hexdigest(),
          'target_sha256':hashlib.sha256(self.docs['base'].raw.encode()).hexdigest(),
          'source_selectors':['line:Изменить','line:Новый адрес','line:Другие условия'],
          'target_selectors':['3.2'],'affected_provisions':['3.2'],
          'applicability':{'field':'specification_date','operator':'>','value':'2024-10-01'}}

    def expand(self,q,primary=None):
        return expand_relations(primary or ['base'],self.docs,[self.rel],q,[self.rule])

    def test_warranty_omits_address_but_compound_question_keeps_it(self):
        order,_,log=self.expand('Какая гарантия по П-17?')
        self.assertEqual(order,['base']);self.assertEqual(log[0]['reason'],'unrelated_topic_skip')
        order,projections,log=self.expand('Какая гарантия и куда доставить партию?')
        self.assertIn('change',order)
        self.assertIn('Новый адрес применяется после даты спецификации.',projections['change'])
        self.assertEqual(log[0]['applicability']['operator'],'>')

    def test_earlier_dates_do_not_erase_amendment_condition(self):
        for day in ('30 сентября','1 октября','2 октября'):
            self.assertIn('change',self.expand(f'Спецификация от {day} 2024 года. Куда доставлять?')[0])

    def test_unknown_broad_and_explicit_amendment_questions_keep_source(self):
        for q in ('Что делать с этим документом?','Какие все изменения договора?','Допсоглашение изменило оплату?'):
            self.assertIn('change',self.expand(q)[0])

    def test_changed_source_disables_rule_and_keeps_full_document(self):
        self.docs['change'].raw+='\nИзменена также гарантия.'
        order,projections,log=self.expand('Какая гарантия?')
        self.assertIn('change',order);self.assertIsNone(projections['change'])
        self.assertEqual(log[0]['reason'],'changed_source_keep')

    def test_direct_reranker_hit_survives_topic_mismatch(self):
        order,projection,_=self.expand('Какая гарантия?',primary=['base','change'])
        self.assertEqual(order,['base','change']);self.assertEqual(projection,{})

    def test_unknown_relation_and_missing_selector_fall_back_to_full_text(self):
        self.rule['source']='unknown.txt'
        self.assertIn('change',self.expand('Какая гарантия?')[0])
        self.rule['source']='change.txt';self.rule['source_selectors']=['line:Отсутствует']
        self.assertIsNone(self.expand('Куда доставить?')[1]['change'])

    def test_context_preserves_verbatim_projection_and_logs_skipped_link(self):
        chunks=[SimpleNamespace(doc_id='base')]
        with patch('relation_scope.rules',return_value=[self.rule]):
            e,b=context_bundle(chunks,self.docs,[self.rel],2000,'Куда доставить?')
            self.assertTrue(all(x['text'] in self.docs['change'].raw for x in e if x['document']=='change'))
            self.assertTrue(any(x['document']=='change' for x in e))
            e,b=context_bundle(chunks,self.docs,[self.rel],2000,'Какая гарантия?')
            self.assertEqual([x['document'] for x in e],['base'])
            self.assertFalse(b['relation_decisions'][0]['keep'])

    def test_retriever_does_not_reintroduce_relation_before_filter(self):
        r=object.__new__(ScopeRetriever)
        r.chunks=[SimpleNamespace(id='a',doc_id='base',parent_id='',ordinal=0),
                  SimpleNamespace(id='b',doc_id='change',parent_id='',ordinal=0)]
        r.all={c.id:c for c in r.chunks};r.docs=self.docs;r.relations=[self.rel]
        self.assertEqual(r.expand(['a'],[0,1]),['a'])


if __name__=='__main__':unittest.main()
