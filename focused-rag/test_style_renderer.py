import unittest
from style_renderer import render_style,STYLES,topic_group

class StyleTests(unittest.TestCase):
    def test_all_modes_preserve_claims_sources_and_temporal_scope(self):
        source='Документ (редакция)'
        answer='Предел — 45 календарных дней. ('+source+', п. 1)\n\nНужно уточнить дату.\n\nПо состоянию на текущую дату.'
        evidence=[{'source_id':'D1','document':source}]
        for style in STYLES:
            with self.subTest(style=style):
                text=render_style(answer,['D1'],evidence,{},style)
                for fragment in ['45 календарных дней',source,'п. 1','Нужно уточнить дату.','текущую дату']:self.assertIn(fragment,text)
                self.assertLessEqual(len(text),10000)

    def test_no_scope_is_invented_and_long_answers_are_preserved(self):
        source=[{'source_id':'D1','document':'Source'}]
        text='На 01.03.2025 нужны 12 рублей. (Source, п. 1)'
        for style in STYLES:self.assertNotIn('текущую дату',render_style(text,['D1'],source,{},style))
        text='Я'*9970+' (Source, п. 1)'
        self.assertEqual(render_style(text,['D1'],source,{},'table_scope'),text)

    def test_compound_topics_keep_default_layout(self):
        self.assertEqual(topic_group(['warranty']),'warranty')
        self.assertIsNone(topic_group(['warranty','contract_term']))

if __name__=='__main__':unittest.main()
