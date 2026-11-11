"""Russian morphology, stable index configuration and original source preservation."""
import os
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from legal_core import LegalAssistant, ModelProxy, search_terms
from ru_stemmer import stem_russian


class StemmingTests(unittest.TestCase):
    def test_inflections_share_stems(self):
        for words, expected in [(('договор', 'договора', 'договору'), 'договор'),
                                (('оплата', 'оплате', 'оплаты'), 'оплат')]:
            for word in words:
                self.assertEqual(stem_russian(word), expected)

    def test_snowball_regions_and_suffix_classes(self):
        examples = {'согласиями': 'соглас', 'подписавшись': 'подписа', 'подписывавшийся': 'подписыва',
                    'работающий': 'работа', 'важнейшими': 'важн', 'длиннейший': 'длин',
                    'радость': 'радост', 'конь': 'кон', 'ва': 'ва', 'а': 'а', 'съёмками': 'съемк'}
        for word, expected in examples.items():
            with self.subTest(word=word):
                self.assertEqual(stem_russian(word), expected)

    def test_numbers_and_identifiers_remain_searchable(self):
        for word in ('2026', '152ФЗ', 'П-17/2024', 'ABC123', 'api'):
            self.assertEqual(stem_russian(word), word.lower())
        self.assertEqual(stem_russian(''), '')
        self.assertEqual(stem_russian('ДОГОВОРА'), 'договор')

    def test_expansion_preserves_originals_and_legacy_mode(self):
        self.assertEqual(search_terms('договора', mode='legacy'), {'договора'})
        self.assertEqual(search_terms('договора', mode='snowball'), {'договора', 'договор'})
        with self.assertRaises(ValueError):
            search_terms('word', mode='unknown')

    def test_original_sources_and_embedding_texts_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'kb.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('contract.txt', 'Договор услуг\n1.1. Срок оплаты договора составляет семь дней.')
            proxy = ModelProxy()
            proxy.base = proxy.key = proxy.rerank_model = ''
            with patch.object(proxy, 'post') as post:
                legacy = LegalAssistant(archive, proxy, use_embeddings=False, lexical_stemmer='legacy')
                snowball = LegalAssistant(archive, proxy, use_embeddings=False, lexical_stemmer='snowball')
            post.assert_not_called()
            self.assertEqual(legacy.sources, snowball.sources)
            self.assertEqual(legacy.source_records, snowball.source_records)
            self.assertEqual(legacy.search_texts, snowball.search_texts)
            self.assertIn('договор', snowball.terms[0])
            # A subsequent environment change must not mismatch the query and index.
            with patch.dict(os.environ, {'LEXICAL_STEMMER': 'legacy'}):
                self.assertIn('договор', snowball._search_terms('договора'))
            with patch.dict(os.environ, {'LEXICAL_STEMMER': 'snowball'}):
                self.assertNotIn('договор', legacy._search_terms('договора'))

    def test_threads_share_only_immutable_cached_results(self):
        words = ['договора', 'оплаты', 'согласиями', 'подписавшись'] * 25
        expected = ['договор', 'оплат', 'соглас', 'подписа'] * 25
        with ThreadPoolExecutor(max_workers=3) as pool:
            self.assertEqual(list(pool.map(stem_russian, words)), expected)


if __name__ == '__main__':
    unittest.main()
