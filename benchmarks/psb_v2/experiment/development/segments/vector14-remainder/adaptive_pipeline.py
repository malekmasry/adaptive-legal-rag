"""Bounded adaptive retrieval, source graph, strict IRAC and fail-closed review.

All stages share the selected lexical retriever. Model-generated plans can only
submit bounded search strings; they cannot choose endpoints or execute tools.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from time import perf_counter

from hallugraph import align, check_temporal, read_claims, read_facts, revision_metadata
from rag_safety import POLICY, clean_data, normalized, parse_object, safe_display, string, strings
from rag_telemetry import CURRENT, Trace


PLAN = """Этап: evidence_plan. Разбери последний вопрос с учетом пользовательской
истории (старые ответы ассистента не доказательство). Выдели до 4 подвопросов.
Выделяй ТОЛЬКО то, что спросил пользователь. Наличие посторонних норм в выдаче
не создаёт новый подвопрос. Проверка условий
должна обслуживать исходный вопрос, а не расширять его на новые темы.
Обстоятельства из вопроса (сумма, дата события) принимай как условия задачи:
не требуй найти саму сделку в базе, если вопрос о применении общего регламента.
Уточнение нужно только когда без него действительно нельзя выбрать норму.
Выбор редакции обосновывай датами и условиями действия из источников. Не считай
дату создания или название файла доказательством начала действия нормы.
Читай положения с учётом раздела, ссылок и исключений; не назначай связанным
положениям общий срок без подтверждения в тексте. Область действия документа
подтверждай фрагментами, а не предполагаемым номером пункта.
Оцени полноту найденных оснований. При конкретном пробеле предложи до 2 поисковых
запросов, сохраняя номер договора и контрагента. Не ищи повторно без причины.
Сначала извлеки независимый граф фактов ИЗ ИСТОЧНИКОВ, а не ответ на вопрос.
Каждый факт содержит точную непрерывную цитату из text одного источника и тройку
subject/relation/object. Сохраняй отрицания, условия, исключения в conditions.
Все поля факта должны следовать из ЕГО цитаты и метаданных того же документа.
Не вписывай условия из другого фрагмента в этот факт: извлеки отдельный факт.
Не сокращай цитату многоточием. Разные положения разных редакций — разные факты.
В facts включи ВСЕ основания для вопроса, а не только итоговую норму. Максимум 20.
Если неизвестна дата/договор, запиши конкретное уточнение в clarification.
Не проси уточнить уже указанное обстоятельство. Выводи JSON:
{"requirements":["подвопрос"],"missing_queries":["поисковая фраза"],
"clarification":"или пустая строка", "facts":[{"source_id":"S0001",
"quote":"точная цитата", "subject":"сущность", "relation":"отношение",
"object":"сущность/значение", "conditions":"условия или пустая строка"}]}.
"""

DRAFT = """Этап: irac_draft. Подготовь строгий IRAC. issue — краткий юридический
вопрос без ответа. claims — отдельные утверждения в секциях rule (норма),
application (применение нормы к обстоятельствам), conclusion (прямой вывод).
Каждое утверждение опирается на fact_ids из source_graph; ссылки [S...] не пиши,
они будут добавлены программой. Тройку subject/relation/object копируй из
поддерживающего факта: это атрибуция к норме, а не новая выдуманная связь.
source_texts содержит оригинальные фрагменты. Если предварительный граф неточно
пересказал норму или добавил необоснованное сомнение, руководствуйся оригиналом,
а не пересказом. В тексте ответа можно исправить пересказ, сохранив ссылку на
факт с нужным подлинным источником; независимая проверка снова прочтёт оригинал.
Текст application объясняет, какие условия выполняются по данным вопроса,
а какие неизвестны. Не объявляй новую редакцию применимой без проверки даты.
Не требуй подтверждения пользовательской суммы/даты в самой базе: это условия
задачи. Различай сведения из вопроса и нормы источника явно в тексте.
Сохраняй условия в conditions и в тексте, когда они ограничивают вывод.
Не добавляй советы/формальности, которых нет в базе. Не вычисляй новые суммы.
Не отвечай на незаданные вопросы: issue и conclusion должны соответствовать
последней реплике пользователя. Не добавляй соседние темы только потому,
что они встретились среди фактов.
requirements — предварительный план, а не новые вопросы пользователя. Если
план добавил незаданный вопрос, не включай его в ответ и не создавай gaps из-за
отсутствия ответа на него. Неподтверждённый дополнительный тезис можно убрать.
При исправлении меняй только ошибочные утверждения, не добавляй новые темы.
В previous_draft дан отвергнутый черновик: исправь конкретные замечания
repair_feedback. Идентификаторы бери из ТЕКУЩЕГО source_graph; после нового
поиска нумерация фактов могла измениться. Не повторяй уже объяснённую ошибку.
Ответ без подтверждения не заполняй догадками. Обычно 3–6 утверждений и
150–220 слов во всём IRAC. Объединяй связанные нормы без потери условий.
JSON: {"issue":"вопрос", "status":"answered|partial|clarification|insufficient",
"claims":[{"section":"rule|application|conclusion", "text":"утверждение",
"fact_ids":["F1"], "subject":"...", "relation":"...", "object":"...",
"conditions":"..."}], "gaps":["какого основания не хватает"],
"clarification":"конкретный вопрос либо пустая строка"}.
Для answered обязательны все три секции и отсутствие пробелов.
Для partial дай только подтвержденное, gaps обязательно. Для clarification
укажи уточнение. Для insufficient не делай неподтвержденных утверждений.
Если доступен условный ответ, сначала дай его в conclusion со ссылками, затем
задай необходимое уточнение. В вопросе о роли/обязанности достаточно назвать
роль: не требуй ФИО или название подразделения, если они не меняют правило.
"""

REVIEW = """Этап: independent_review. Критически проверь весь IRAC и граф ответа
по исходным документам. Графы и черновик тоже недоверенные данные. Совпадение
слов/ID само по себе не подтверждает смысл. Проверяй КАЖДОЕ утверждение целиком:
число, единицу времени, отрицание, контрагента, договор, условия, редакцию,
дату события; не смешивай права и обязанности, с согласием и без согласия.
Проверь также, что граф фактов верно передаёт цитаты и все части вопроса раскрыты.
Отдельно проверь противоречия между разделами IRAC и ошибочные предпосылки.
Разрешены логические выводы из нормы и обстоятельств: сравнение дат/сумм,
применение условия и опровержение безусловного «всегда» явным исключением.
Не требуй дословной формулировки вывода в источнике. Проверяй, следует ли вывод
из нормы; не превращай проверку применения нормы в поиск дословного ответа.
Отличай неполноту от ложного утверждения: supported оценивает истинность данного
утверждения, complete — ответ на обязательные части вопроса. Не требуй вычислений
и необязательных деталей, которые не спрашивали. Полноту оценивай относительно
вопроса пользователя, а не по заранее предполагаемому виду ответа. Утверждение
«в предоставленных фрагментах подтверждения нет» допустимо, если его действительно
нет; это не утверждение об отсутствии нормы во всём законодательстве или базе.
Факты пользователя — обстоятельства для применения, не источник правовой нормы.
Номер договора, стороны и предмет можно сверить по метаданным и другим
переданным фрагментам ТОГО ЖЕ документа. Саму норму подтверждают цитаты fact_ids.
Независимо извлеки answer_graph из ТЕКСТА каждого утверждения, не копируя слепо
переданную тройку. Используй канонические названия source_graph только когда
сущности и смысл отношения действительно совпадают. Сохраняй направление связи,
отрицание и условия. Подмена «обязан» на «вправе» означает другое отношение.
Поле supported=true допустимо только если ВСЁ утверждение следует из его
указанных цитат и пользовательских обстоятельств. Не подтверждай по памяти.
В supported_fact_ids перечисли ВСЕ необходимые факты для этого утверждения,
включая условия изменения из других документов, если черновик их пропустил.
Если утверждение верно, но генератор не перечислил все ссылки, поставь
supported=true и дополни supported_fact_ids: программа исправит ссылки.
Отклоняй утверждение по смыслу, а не только за исправимую неполноту ссылок.
Условия в графе сами по себе не доказательство: проверь их по исходным цитатам.
Если полного доказательства нет, supported=false, не подбирай похожий факт.
Для каждой явной проверки даты запиши temporal_checks. Строгое «после» — after,
«с» — from, строгое «до» — before, «включительно/не позднее» — through.
Если дата события неизвестна, не придумывай её, отклони безусловное применение.
operator описывает границу действия НОРМЫ из цитаты: при «с» используй from,
даже если событие произошло позже границы, а не after по слову из черновика.
В temporal_checks включай только полные известные календарные даты; пустые
даты запрещены. Когда указан лишь месяц/год, проверь интервал в reason.
JSON: {"verdicts":[{"claim_id":"C1","supported":true,"reason":"кратко",
"supported_fact_ids":["F1"]}],
"answer_graph":[{"claim_id":"C1","subject":"...","relation":"...",
"object":"...","conditions":"..."}],
"complete":true, "consistent":true, "status_supported":true,
"issue_supported":true, "gaps_supported":true, "clarification_supported":true,
"missing_queries":["до 2 фраз для поиска недостающего основания"],
"temporal_checks":[{"event_date":"YYYY-MM-DD","boundary":"YYYY-MM-DD",
"operator":"after|from|before|through", "applies":false,
"source_id":"S0001","quote":"точная цитата условия"}]}.
verdicts должен покрывать все claims ровно один раз. В answer_graph для каждого
claim дай от одной до четырёх троек, отражающих все его существенные отношения.
complete=false если не
раскрыт хотя бы один подвопрос. status_supported проверяет обоснованность
ответа/отказа/частичного ответа/уточнения. Для partial complete может быть false.
"""


def bounded_queries(value) -> list[str]:
    result = []
    for query in strings(value, 2, 500):
        cleaned, _ = clean_data(query)
        if cleaned and safe_display(cleaned):
            result.append(cleaned)
    return result


class AdaptivePipeline:
    def __init__(self, engine):
        self.engine = engine
        self.trace = Trace()
        self.searched: set[str] = set()
        self.rounds = 0
        self.calls = 0
        self.evidence: dict[str, dict] = {}
        self.facts = []
        self.graph = {}

    def call(self, stage: str, instructions: str, payload: dict) -> dict:
        from legal_core import integer_setting
        if self.calls >= integer_setting("MAX_PIPELINE_CALLS", 7, 3, 10) or self.trace.remaining() < 1:
            raise RuntimeError("Pipeline budget exhausted")
        self.calls += 1
        self.trace.stage = stage
        started = perf_counter()
        try:
            response = self.engine.proxy.complete(POLICY + instructions, json.dumps(payload, ensure_ascii=False))
            return parse_object(response)
        finally:
            self.trace.timings[stage] = self.trace.timings.get(stage, 0) + (perf_counter() - started) * 1000

    def retrieve(self, query: str, reason: str) -> list:
        key = normalized(query)
        if key in self.searched:
            return []
        self.searched.add(key)
        started = perf_counter()
        found = self.engine.retrieve(query)
        self.trace.timings["retrieval"] = self.trace.timings.get("retrieval", 0) + (perf_counter() - started) * 1000
        self.trace.events.append({"stage": "retrieval", "reason": reason,
                                  "source_ids": [source.id for source in found]})
        return found

    def pack(self, candidates: list) -> None:
        from legal_core import integer_setting
        started = perf_counter()
        max_chars = integer_setting("CONTEXT_MAX_CHARS", 30000, 8000, 80000)
        max_sources = integer_setting("CONTEXT_MAX_SOURCES", 24, 12, 48)
        seen, selected, total = set(), {}, 0
        documents = {doc.id: doc for doc in self.engine.documents}
        for source in candidates:
            if source.id in selected:
                continue
            text, removed = clean_data(source.text)
            if removed:
                self.trace.events.append({"stage": "quarantine", "source_id": source.id, "lines": removed})
            key = (source.document_id, normalized(text))
            if not text or key in seen:
                continue
            title, _ = clean_data(source.document)
            path, _ = clean_data(source.path)
            doc = documents[source.document_id]
            clean_document, _ = clean_data(doc.text)
            # Preserve actual surrounding headings and source offsets.
            before = doc.text[:source.start].splitlines()
            headings = [line.strip() for line in before if line.strip() and
                        (re.match(r"^(?:Статья|Раздел|Глава)\s|^\d+\.\s", line.strip()) or
                         line.rstrip().endswith(":"))]
            scope, _ = clean_data("\n".join(headings[-2:]))
            record = {"id": source.id, "document": title, "path": path,
                      "document_id": source.document_id,
                      "clause": source.clause, "text": text, "parent_scope": scope[-600:],
                      "contract_number": doc.contract_number}
            record["revision"] = revision_metadata(clean_document)
            record["parties"] = " ".join(f"«{name}»" for name in re.findall(r"«([^»]+)»", clean_document[:1200])[:10])
            size = len(json.dumps(record, ensure_ascii=False))
            if total + size > max_chars or len(selected) >= max_sources:
                continue
            seen.add(key)
            selected[source.id] = record
            total += size
        self.evidence = selected
        self.trace.events.append({"stage": "context", "characters": total, "sources": len(selected)})
        self.trace.timings["context_pack"] = self.trace.timings.get("context_pack", 0) + (perf_counter() - started) * 1000

    def enrich(self, candidates: list) -> list:
        """Include nearby continuation/list scope; no new graph traversal."""
        started = perf_counter()
        referenced = []
        seen_clauses = set()
        for source in candidates[:6]:
            permitted_documents = {source.document_id}
            for link in self.engine.links:
                if source.document_id in {link.source_document_id, link.target_document_id}:
                    permitted_documents.update((link.source_document_id, link.target_document_id))
            references = re.findall(r"(?:пункт[а-я]*|п\.)\s*(\d+\.\d+(?:\.\d+)?)", source.text, re.I)
            for clause in references:
                reference_key = (source.document_id, clause)
                if reference_key in seen_clauses or len(referenced) >= 6:
                    continue
                seen_clauses.add(reference_key)
                matching = {index for index, candidate in enumerate(self.engine.sources)
                            if candidate.clause == clause and candidate.id != source.id
                            and candidate.document_id in permitted_documents}
                if matching:
                    ranked = self.engine._lexical(source.text)
                    additions = [self.engine.sources[index] for index in ranked if index in matching][:2]
                    referenced.extend(additions)
                    self.trace.events.append({"stage": "explicit_reference", "from": source.id,
                                              "clause": clause, "to": [item.id for item in additions]})
        expanded = candidates[:8] + referenced + candidates[8:]
        for source in candidates[:12]:
            siblings = sorted(self.engine.by_document[source.document_id], key=lambda item: item.start)
            position = next(index for index, item in enumerate(siblings) if item.id == source.id)
            for index in (position - 1, position + 1):
                if 0 <= index < len(siblings):
                    neighbor = siblings[index]
                    if neighbor.clause == source.clause or neighbor.text.rstrip().endswith(":"):
                        expanded.append(neighbor)
        self.trace.timings["reference_and_neighbors"] = self.trace.timings.get("reference_and_neighbors", 0) + (perf_counter() - started) * 1000
        return expanded

    def expand(self, queries: list[str], candidates: list, question: str) -> list:
        from legal_core import CONTRACT_RE, integer_setting
        if self.rounds >= integer_setting("MAX_RETRIEVAL_ROUNDS", 1, 0, 2):
            return candidates
        queries = [query for query in queries if normalized(query) not in self.searched]
        if not queries:
            return candidates
        self.rounds += 1
        extra = []
        identifiers = CONTRACT_RE.findall(question)
        for query in queries:
            # Preserve exact contract scope omitted by the planner.
            query = query + " " + " ".join(number for number in identifiers if number not in query)
            extra.extend(self.retrieve(query.strip(), "missing_evidence"))
        # Reserve space for targeted discoveries; append original tail afterwards.
        return candidates[:8] + extra[:16] + candidates[8:] + extra[16:]

    def source_plan(self, question: str) -> dict:
        plan = self.call("evidence_plan", PLAN, {"question": question, "sources": list(self.evidence.values())})
        requirements = strings(plan.get("requirements"), 4, 500)
        if not requirements or not all(safe_display(value) for value in requirements):
            raise ValueError("Invalid requirements")
        self.facts = read_facts(plan.get("facts"), self.evidence)
        return {"requirements": requirements, "missing_queries": bounded_queries(plan.get("missing_queries")),
                "clarification": string(plan.get("clarification"), 500)}

    def validate_draft(self, draft: dict):
        draft["issue"] = string(draft.get("issue"), 600)
        draft["gaps"] = strings(draft.get("gaps"), 8, 500)
        draft["clarification"] = string(draft.get("clarification"), 500)
        if draft.get("status") not in {"answered", "partial", "clarification", "insufficient"}:
            raise ValueError("Invalid answer status")
        if not draft["issue"] or not all(safe_display(value) for value in
                                          [draft["issue"], draft["clarification"]] + draft["gaps"]):
            raise ValueError("Unsafe narrative")
        claims = read_claims(draft.get("claims"), self.facts)
        if draft["status"] == "answered" and ({claim.section for claim in claims} != {"rule", "application", "conclusion"}
                                               or draft["gaps"] or draft["clarification"]):
            raise ValueError("Incomplete IRAC")
        if draft["status"] == "partial" and (not claims or not draft["gaps"]):
            raise ValueError("Invalid partial answer")
        if draft["status"] == "clarification" and not draft["clarification"]:
            raise ValueError("Missing clarification")
        return claims

    def review(self, question: str, plan: dict, draft: dict, claims: list) -> tuple[bool, dict, list]:
        review = self.call("verification", REVIEW, {
            "question": question, "requirements": plan["requirements"],
            "sources": list(self.evidence.values()), "source_graph": [fact.public() for fact in self.facts],
            "draft": {**draft, "claims": [claim.public() for claim in claims]},
        })
        checks = review.get("verdicts")
        if not isinstance(checks, list) or len(checks) != len(claims):
            raise ValueError("Incomplete review")
        ids = [item.get("claim_id") for item in checks if isinstance(item, dict)]
        if len(ids) != len(set(ids)) or set(ids) != {claim.id for claim in claims}:
            raise ValueError("Invalid review claim IDs")
        for field in ("complete", "consistent", "status_supported", "issue_supported", "gaps_supported", "clarification_supported"):
            if type(review.get(field)) is not bool:
                raise ValueError("Invalid review flag")
        for item in checks:
            if type(item.get("supported")) is not bool:
                raise ValueError("Invalid support flag")
            string(item.get("reason"), 1200)
            refs = strings(item.get("supported_fact_ids"), 12, 16)
            if not set(refs) <= {fact.id for fact in self.facts} or (item["supported"] and not refs):
                raise ValueError("Invalid semantic proof references")
        proof = {item["claim_id"]: item["supported_fact_ids"] for item in checks}
        claims = [replace(claim, fact_ids=tuple(dict.fromkeys(claim.fact_ids + tuple(proof[claim.id])))) for claim in claims]
        triples = review.get("answer_graph")
        if not isinstance(triples, list) or not len(claims) <= len(triples) <= len(claims) * 4:
            raise ValueError("Incomplete independently extracted answer graph")
        indexed = {}
        for item in triples:
            claim_id = string(item.get("claim_id"), 16)
            triple = {key: string(item.get(key), 500) for key in
                      ("subject", "relation", "object", "conditions")}
            indexed.setdefault(claim_id, [])
            if triple not in indexed[claim_id]:
                indexed[claim_id].append(triple)
            if len(indexed[claim_id]) > 4:
                raise ValueError("Too many claim relations")
        if set(indexed) != {claim.id for claim in claims}:
            raise ValueError("Answer graph claim mismatch")
        extracted = [replace(claim, **triple) for claim in claims for triple in indexed[claim.id]]
        self.graph = align(self.facts, extracted, self.evidence, question)
        # OpenIE phrases and valid applications of conditional rules need not be
        # literally identical to source edges. Keep literal scores for audit;
        # semantic alignment requires an explicit independently reviewed proof.
        self.graph["semantic_relation_preservation"] = (sum(item["supported"] for item in checks) / len(checks)
                                                        if checks else None)
        self.graph["semantic_alignments"] = [{"claim_id": item["claim_id"], "fact_ids": item["supported_fact_ids"],
                                              "supported": item["supported"]} for item in checks]
        hard_issues = {"unsupported_number", "unsupported_contract", "unsupported_named_entity",
                       "reversed_relation", "relation_polarity_changed", "inclusive_exclusive_conflict"}
        hard_findings = [item for item in self.graph["findings"] if hard_issues.intersection(item["issues"])]
        temporal = check_temporal(review.get("temporal_checks"), question, self.evidence)
        queries = bounded_queries(review.get("missing_queries"))
        accepted = (not hard_findings and not temporal and all(item["supported"] for item in checks)
                    and all(review[field] for field in ("consistent", "status_supported", "issue_supported",
                                                       "gaps_supported", "clarification_supported"))
                    and (review["complete"] or draft["status"] != "answered"))
        details = {"verdicts": checks, "complete": review["complete"], "consistent": review["consistent"],
                   "temporal_issues": temporal, "missing_queries": queries,
                   "graph_findings": self.graph["findings"], "accepted": accepted}
        details.update({key: review[key] for key in ("status_supported", "issue_supported", "gaps_supported", "clarification_supported")})
        self.trace.events.append({"stage": "verification", **details})
        return accepted, details, claims

    def result(self, text: str, sources: list, status: str):
        from legal_core import Answer
        diagnostics = self.trace.public()
        diagnostics["search_total_ms"] = round(self.trace.timings.get("retrieval", 0) +
                                                self.trace.timings.get("reference_and_neighbors", 0), 2)
        diagnostics.update({"retrieval_rounds": self.rounds, "pipeline_calls": self.calls,
                            "ingestion_warnings": self.engine.ingestion_warnings,
                            "hallugraph": self.graph, "retriever": "ours_lexical" if self.engine.vectors is None else "ours_hybrid"})
        return Answer(text, sources, status, diagnostics)

    def fallback(self, status: str):
        if status == "technical_error":
            text = ("I — Вопрос\nНе удалось завершить обработку вопроса.\n\n"
                    "R — Нормы\nПроверка источников не завершена.\n\n"
                    "A — Применение\nПодтверждённый анализ недоступен из-за технической ошибки.\n\n"
                    "C — Вывод\nПовторите запрос позже.")
        else:
            text = ("I — Вопрос\nВопрос по предоставленной базе документов.\n\n"
                    "R — Нормы\nНадёжно подтверждённые основания не установлены.\n\n"
                    "A — Применение\nПроверка не позволила подтвердить вывод с учётом условий вопроса.\n\n"
                    "C — Вывод\nНедостаточно подтверждённых оснований для ответа. "
                    "Это не означает, что соответствующая норма отсутствует во всей базе.")
        return self.result(text, [], status)

    def render(self, draft: dict, claims: list):
        facts = {fact.id: fact for fact in self.facts}
        cited = []
        sections = {name: [] for name in ("rule", "application", "conclusion")}
        for claim in claims:
            source_ids = list(dict.fromkeys(facts[ref].source_id for ref in claim.fact_ids))
            sections[claim.section].append(claim.text + " " + " ".join(f"[{sid}]" for sid in source_ids))
            cited.extend(source_ids)
        if draft["gaps"]:
            sections["application"].append("Не хватает оснований: " + "; ".join(draft["gaps"]) + ".")
        if draft["clarification"]:
            sections["conclusion"].append(draft["clarification"])
        empty = {"rule": "Подтверждённая норма для полного ответа не установлена.",
                 "application": "Достаточных оснований для применения нормы нет.",
                 "conclusion": "Недостаточно подтверждённых оснований для полного ответа."}
        text = "I — Вопрос\n" + draft["issue"]
        for key, label in (("rule", "R — Нормы"), ("application", "A — Применение"), ("conclusion", "C — Вывод")):
            text += "\n\n" + label + "\n" + ("\n".join(sections[key]) or empty[key])
        return self.result(text, [self.engine.by_id[sid] for sid in dict.fromkeys(cited)], draft["status"])

    def run(self, question: str):
        from legal_core import focused_search_question, integer_setting
        self.trace.deadline_seconds = integer_setting("PIPELINE_TIMEOUT_SECONDS", 180, 20, 300)
        token = CURRENT.set(self.trace)
        try:
            cleaned, removed = clean_data(question)
            self.trace.events.append({"stage": "input", "quarantined_lines": removed})
            if not cleaned.strip():
                return self.result("I — Вопрос\nВопрос по базе не определён.\n\nR — Нормы\nНе выбраны.\n\n"
                                   "A — Применение\nНедостаточно обстоятельств.\n\nC — Вывод\n"
                                   "Уточните, пожалуйста, юридический вопрос по документам.", [], "clarification")
            if len(question) > 20000:
                raise ValueError("Question too long")
            if not self.engine.proxy.ready:
                return self.fallback("technical_error")
            candidates = self.retrieve(focused_search_question(cleaned), "initial")
            self.pack(self.enrich(candidates))
            if not self.evidence:
                return self.fallback("insufficient")
            plan = self.source_plan(cleaned)
            if plan["missing_queries"]:
                candidates = self.expand(plan["missing_queries"], candidates, focused_search_question(cleaned))
                self.pack(self.enrich(candidates))
                plan = self.source_plan(cleaned)
            feedback = None
            previous_draft = None
            for attempt in range(2):
                payload = {"question": cleaned, "requirements": plan["requirements"],
                           "clarification_needed": plan["clarification"],
                           "source_graph": [fact.public() for fact in self.facts], "repair_feedback": feedback,
                           "previous_draft": previous_draft}
                payload["source_metadata"] = [{key: item[key] for key in
                    ("id", "document", "document_id", "path", "contract_number", "parties", "revision")}
                    for item in self.evidence.values()]
                relevant_ids = {fact.source_id for fact in self.facts}
                payload["source_texts"] = [item for sid, item in self.evidence.items() if sid in relevant_ids]
                try:
                    draft = self.call("generation" if attempt == 0 else "repair", DRAFT, payload)
                    previous_draft = draft
                    claims = self.validate_draft(draft)
                    accepted, feedback, claims = self.review(cleaned, plan, draft, claims)
                except (ValueError, KeyError, TypeError) as exc:
                    if attempt:
                        raise
                    feedback = {"schema_error": str(exc)[:180], "missing_queries": [],
                                "instruction": "Соблюди схему IRAC и перепроверь обоснованность ответа/отказа по графу и метаданным."}
                    self.trace.events.append({"stage": "schema_repair", "type": type(exc).__name__})
                    continue
                if accepted:
                    return self.render(draft, claims)
                if attempt == 0 and feedback["missing_queries"] and self.rounds < integer_setting("MAX_RETRIEVAL_ROUNDS", 1, 0, 2):
                    candidates = self.expand(feedback["missing_queries"], candidates, focused_search_question(cleaned))
                    self.pack(self.enrich(candidates))
                    plan = self.source_plan(cleaned)
            return self.fallback("insufficient")
        except (RuntimeError, KeyError, IndexError, TypeError, ValueError, RecursionError) as exc:
            self.trace.events.append({"stage": "failure", "type": type(exc).__name__})
            return self.fallback("technical_error")
        finally:
            CURRENT.reset(token)
