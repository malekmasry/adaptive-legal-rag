"""Content-driven behavior under changed companies, years, filenames and clauses.

These tests exercise ingestion/retrieval/reference resolution without a question
bank or API. They do not measure end-to-end answer quality on an unseen corpus.
"""
from __future__ import annotations

import random
import tempfile
import unittest
import zipfile
from pathlib import Path

from adaptive_pipeline import AdaptivePipeline
from legal_core import Document, LegalAssistant, build_links, search_terms


class OfflineProxy:
    base = key = rerank_model = ""


def document(identifier, title, body, *, path=None, contract=None):
    return Document(identifier, path or identifier + ".txt", title, title + "\n" + body, "company", contract)


class CorpusIndependenceTests(unittest.TestCase):
    def engine(self, documents):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        archive = Path(temporary.name) / "corpus.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path, text in documents.items():
                bundle.writestr(path, text)
        return LegalAssistant(archive, OfflineProxy(), use_embeddings=False)

    def test_amendments_follow_content_identifiers_under_renaming(self):
        for seed in (19, 71, 113):
            rng = random.Random(seed)
            number = f"Ц-{rng.randrange(30, 99)}/{rng.randrange(1990, 2040)}"
            title = "Договор обслуживания датчиков № " + number
            original = document("base", title, "9.3. Замена сенсора по заявке.", path="random_a.txt", contract=number)
            change = document("change", "Дополнительное соглашение", "к договору № " + number + "\nНовый порядок замены.", path="random_b.txt", contract=number)
            links = build_links([original, change])
            self.assertEqual([(link.source_document_id, link.target_document_id, link.relation) for link in links],
                             [("change", "base", "amends")])
            self.assertIn(number, links[0].basis)

    def test_years_and_legacy_filenames_cannot_create_supersession(self):
        docs = [document("old", "Порядок калибровки датчиков", "Редакция от 12.02.1998\n7.2. Калибровка ежемесячно.",
                         path="регламент согласования договоров ред 2023.txt"),
                document("new", "Порядок калибровки датчиков", "Редакция от 18.07.2037\n7.2. Калибровка еженедельно.",
                         path="регламент согласования договоров ред 2025.txt")]
        self.assertEqual(build_links(docs), [])

    def test_explicit_reference_selects_edition_from_text(self):
        for year in (1996, 2017, 2038):
            title = "Правила обслуживания измерителей"
            docs = [document("old", title, f"Редакция от 11.02.{year-1}\n4.3. Диагностика.", path="new-looking.txt"),
                    document("selected", title, f"Редакция от 11.02.{year}\n6.8. Диагностика.", path="old-looking.txt"),
                    document("letter", "Уведомление отдела метрологии", f"Применяется документ «{title}», редакция от 11.02.{year}.")]
            links = build_links(docs)
            self.assertEqual([(link.source_document_id, link.target_document_id, link.relation) for link in links],
                             [("letter", "selected", "references")])
            self.assertIn(str(year), links[0].basis)

    def test_abbreviated_reference_resolves_edition_and_clause(self):
        for title, reference, year, clause in (("Правила измерения токсичности", "Правил измерения токсичности", 2037, "6.8"),
                                               ("Положение о доступе в лабораторию", "Положения о доступе в лабораторию", 1996, "12.4")):
            old = f"{title.upper()} ООО «Лаборатория Фотон»\nРедакция от 11.02.{year-2}.\n{clause}. Прежняя процедура."
            current = f"{title.upper()} ООО «Лаборатория Фотон»\nРедакция от 11.02.{year}.\n{clause}. Проверку выполняет специалист контроля."
            cited = f"СЛУЖЕБНОЕ ПИСЬМО\nПрименяется п. {clause} {reference} (ред. от 11.02.{year})."
            engine = self.engine({"version_x.txt": old, "version_y.txt": current, "notice.txt": cited})
            notice = next(doc for doc in engine.documents if doc.path == "notice.txt")
            target = next(doc for doc in engine.documents if doc.path == "version_y.txt")
            matches = [link for link in engine.links if link.source_document_id == notice.id]
            self.assertEqual([(link.target_document_id, link.target_clause) for link in matches], [(target.id, clause)])
            found = engine.retrieve("Какая процедура указана в служебном письме?")
            self.assertTrue(any(item.document_id == target.id and item.clause == clause for item in found))

    def test_ambiguous_reference_does_not_choose_newest_edition(self):
        title = "Политика обращения с образцами"
        docs = [document("one", title, "Редакция от 12.02.2011"),
                document("two", title, "Редакция от 12.02.2031"),
                document("letter", "Уведомление лаборатории", f"См. документ «{title}».")]
        self.assertEqual(build_links(docs), [])

    def test_same_contract_number_in_two_documents_is_not_unique_identity(self):
        docs = [document("one", "Договор № Р-81/2034", "ООО «Кварц»", contract="Р-81/2034"),
                document("two", "Договор № Р-81/2034", "ООО «Призма»", contract="Р-81/2034"),
                document("change", "Дополнительное соглашение", "к договору № Р-81/2034", contract="Р-81/2034")]
        self.assertEqual(build_links(docs), [])

    def test_self_reference_is_not_a_reference_to_another_edition(self):
        title = "Порядок проверки оборудования"
        docs = [document("one", title, f"Редакция от 12.02.2011\nДокумент «{title}» обязателен."),
                document("two", title, f"Редакция от 12.02.2031\nДокумент «{title}» обязателен.")]
        self.assertEqual(build_links(docs), [])

    def test_retrieval_uses_company_and_contract_inside_arbitrary_files(self):
        for company, other, year, clause, delay in (("Кварц", "Призма", 2032, "7.8", 13),
                                                   ("Маяк", "Рубин", 1999, "12.4", 29)):
            wanted = f"Договор обслуживания № Ц-61/{year}\nООО «{company}»\n{clause}. Срок замены кассеты — {delay} дней."
            distractor = f"Договор обслуживания № Ц-62/{year}\nООО «{other}»\n{clause}. Срок замены кассеты — 41 день."
            for paths in (("a.txt", "z.txt"), ("законы/z.txt", "a.txt")):
                engine = self.engine(dict(zip(paths, (wanted, distractor))))
                for query in (f"Срок замены кассеты по Ц-61/{year}?", f"Срок замены кассеты у «{company}»?"):
                    top = engine.sources[engine._lexical(query)[0]]
                    self.assertIn(str(delay), top.text)
                    self.assertEqual(top.clause, clause)
                self.assertTrue(all(doc.kind == "company" for doc in engine.documents))

    def test_subject_clause_is_retrieved_at_arbitrary_number(self):
        for clause in ("4.7", "9.2", "15.6"):
            engine = self.engine({"unrelated_name.txt": "Договор сервисного сопровождения № Ф-72/2031\n"
                                  "1.1. Документы хранятся в архиве.\n"
                                  f"{clause}. Обслуживаемые устройства — газовые анализаторы.\n"
                                  "18.2. Периодичность проверки — раз в квартал."})
            found = engine.retrieve("Какие устройства обслуживаются по договору Ф-72/2031?")
            self.assertEqual(found[0].clause, clause)
            pipeline = AdaptivePipeline(engine)
            pipeline.pack(found)
            self.assertTrue(any("газовые анализаторы" in value["text"] for value in pipeline.evidence.values()))
            self.assertTrue(all("document_scope" not in value for value in pipeline.evidence.values()))

    def test_clause_number_alone_cannot_link_unrelated_documents(self):
        engine = self.engine({"first.txt": "Договор обслуживания № М-37/2032\n"
                              "2.3. Порядок возврата указан в пункте 8.7.\n8.7. Возврат датчика оформляется актом.",
                              "second.txt": "Договор перевозки № М-38/2032\n"
                              "8.7. Порядок возврата контейнера — предварительная заявка."})
        origin = next(source for source in engine.sources if source.clause == "2.3")
        pipeline = AdaptivePipeline(engine)
        pipeline.enrich([origin])
        events = [event for event in pipeline.trace.events if event["stage"] == "explicit_reference"]
        self.assertTrue(events)
        for event in events:
            self.assertTrue(all(engine.by_id[sid].document_id == origin.document_id for sid in event["to"]))

    def test_query_expansion_contains_only_word_forms(self):
        for word in ("грузовик", "ночью", "испытательный", "измерениями", "кассеты"):
            self.assertTrue(all(word.startswith(term) for term in search_terms(word)))

    def test_inflected_quoted_company_name_matches_without_merging_similar_names(self):
        for name, inflected, other in (("Лазурный Архив", "Лазурного Архива", "Лазурный Архивариус"),
                                       ("Кварц-Трейдинг", "Кварц-Трейдингу", "Кварц-Трейд")):
            engine = self.engine({"random_a.txt": f"Договор хранения\nООО «{name}»\n"
                                  "7.3. По заявке клиента ячейка предоставляется на семь дней.",
                                  "random_b.txt": f"Договор хранения\nООО «{other}»\n"
                                  "7.3. По заявке клиента ячейка предоставляется на сорок дней."})
            ranked = engine.retrieve(f"На сколько дней предоставляется ячейка у «{inflected}»?")
            self.assertIn("семь дней", ranked[0].text)
            self.assertNotIn("сорок дней", ranked[0].text)


if __name__ == "__main__":
    unittest.main()
