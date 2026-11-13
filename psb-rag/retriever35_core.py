"""Retrieval-only donor: dev_if_test 5d7f10d193edfca14f8c94a7f343e4f70303cc6a.
Ranking, splitting, relation expansion and record packing retain donor semantics.
Generation, answer checking and HTTP transport are supplied by v8, not this module.
"""
from __future__ import annotations
import json, math, os, re, time, unicodedata
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
CURRENT = ContextVar('retriever35_trace', default=None)

ATTACK_LINE = re.compile(
    r"для\s+(?:ии|ai)[- ]?ассистента|"
    r"(?:ignore|disregard|forget).{0,50}(?:instructions|system|rules)|"
    r"(?:игнорир|забудь|отменяй|обойди).{0,60}(?:инструкц|правил|документ|промпт)|"
    r"(?:раскрой|покажи|выведи|отправь).{0,60}(?:системн.{0,12}промпт|api.?ключ|секрет|парол)|"
    r"(?:system|developer)\s*:|<\|(?:im_start|im_end|system|assistant)|"
    r"\[/?INST\]|</?(?:system|developer)>|"
    r"(?:системн\w*\s+(?:промпт|инструкц)|новая\s+роль|jailbreak)", re.I,
)


INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def clean_data(text: str) -> tuple[str, int]:
    """Heuristic quarantine, NOT a complete injection detector. Keep originals elsewhere."""
    value = INVISIBLE.sub("", unicodedata.normalize("NFKC", text)).replace("\x00", "")
    value = re.sub(r"\b(?:gw_sk_[A-Za-z0-9_-]{16,}|KGAT_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{20,})\b",
                   "[REDACTED_CREDENTIAL]", value)
    lines, removed = [], 0
    for line in value.splitlines():
        if ATTACK_LINE.search(line):
            removed += 1
        else:
            lines.append(line)
    return "\n".join(lines), removed

MONTHS = {name: index for index, name in enumerate(
    ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
     "сентября", "октября", "ноября", "декабря"), 1)}


DATE_PATTERN = r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b"


def named_entity_key(value: str) -> str:
    """Narrow Russian case normalization, not fuzzy company-name matching."""
    suffixes = ("ого", "ему", "ому", "ыми", "ими", "ами", "ями", "ый", "ий", "ой",
                "ая", "яя", "ое", "ее", "ые", "ие", "ом", "ем", "ым", "им", "ам",
                "ям", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е")
    def stem(match):
        word = match[0]
        if word.endswith(("ие", "ия")) and len(word) > 5:
            return word[:-2] + "и"
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[:-len(suffix)]
        return word
    return re.sub(r"[а-я]+", stem, normalized(value))


def dates(text: str) -> set[date]:
    result = set()
    candidates = [(int(y), MONTHS[m], int(d)) for d, m, y in re.findall(DATE_PATTERN, text.lower())]
    candidates += [(int(y), int(m), int(d)) for d, m, y in re.findall(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", text)]
    candidates += [(int(y), int(m), int(d)) for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)]
    for year, month, day in candidates:
        try:
            result.add(date(year, month, day))
        except ValueError:
            pass
    return result


def revision_metadata(text: str) -> dict[str, str]:
    """Extract labeled dates; never equate signing, revision and effective dates."""
    header = text[:1800]
    patterns = {"revision_date": r"редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})",
                "effective_from": r"(?:действует|вступает\s+в\s+силу)\s+с\s+(\d{2}\.\d{2}\.\d{4})",
                "effective_until_exclusive": r"утратила\s+силу\s+с\s+(\d{2}\.\d{2}\.\d{4})"}
    result = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, header, re.I)
        if match and dates(match.group(1)):
            result[name] = match.group(1)
    return result

WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{2,}")


CLAUSE_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.\s+")


CONTRACT_RE = re.compile(r"(?:№\s*)?([А-ЯA-Z]{1,4}[-–]\d+[/\-]\d{4})", re.I)


def tokens(value: str) -> list[str]:
    return [x.lower().replace("ё", "е") for x in WORD_RE.findall(value)]


def search_terms(value: str) -> set[str]:
    """Language-level morphology only; no topic-specific synonym dictionary."""
    words = set(tokens(value))
    expanded = set(words)
    for word in words:
        if len(word) >= 5:
            for suffix in ("ого", "ему", "ыми", "ими", "ами", "ями", "ой", "ый", "ий", "ая", "ое", "ом", "ем", "ым", "им", "ам", "ям", "ах", "ях"):
                if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                    expanded.add(word[:-len(suffix)])
                    break
    return expanded


def character_trigrams(value: str) -> set[str]:
    """Match inflected Russian words without a language-specific dependency."""
    return {value[index:index + 3] for index in range(len(value) - 2)}


def quoted_names(value: str) -> list[str]:
    return [a or b for a, b in re.findall(r'«([^»]+)»|"([^"\n]+)"', value)]


def latest_turn(question: str) -> str:
    return question.rsplit("Пользователь:", 1)[-1].strip() if "Пользователь:" in question else question


def focused_search_question(question: str) -> str:
    latest = latest_turn(question)
    # When the user resolved the company or contract in the last turn,
    # earlier assistant guesses should not compete with that entity.
    if latest != question and (CONTRACT_RE.search(latest) or re.search(r"[«\"][^»\"]+[»\"]", latest)):
        return latest
    if latest != question:
        user_turns = [turn.strip() for turn in re.findall(
            r"Пользователь:\s*(.*?)(?=\n(?:Пользователь|Ассистент):|\Z)",
            question,
            flags=re.S,
        )]
        if user_turns:
            # An assistant's earlier answer is not evidence about the user's
            # intended document and can drown out the latest question.
            earlier = "\n".join(user_turns[:-1])
            identifiers = CONTRACT_RE.findall(earlier)
            entities = quoted_names(earlier)
            if identifiers or entities:
                return latest + " " + " ".join(identifiers + [f"«{name}»" for name in entities])
            return "\n".join(user_turns)
    return question


def setting(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def integer_setting(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(setting(name))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Document:
    id: str
    path: str
    title: str
    text: str
    kind: str
    contract_number: str | None


@dataclass(frozen=True)
class Source:
    id: str
    document_id: str
    document: str
    path: str
    clause: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Link:
    source_document_id: str
    target_document_id: str
    relation: str
    basis: str
    target_clause: str | None = None


def split_sources(documents: list[Document], limit: int = 1400) -> list[Source]:
    """Split at source line/paragraph boundaries; keep offsets for exact display."""
    units: list[tuple[Document, str, str, int, int]] = []
    for doc in documents:
        starts = [0]
        for match in re.finditer(r"\n", doc.text):
            starts.append(match.end())
        boundaries = []
        for start in starts:
            line = doc.text[start:doc.text.find("\n", start) if "\n" in doc.text[start:] else len(doc.text)]
            found = CLAUSE_RE.match(line)
            if found:
                boundaries.append((start, found.group(1)))
        articles = list(re.finditer(r"(?m)^\s*Статья\s+(\d+(?:\.\d+)*)[.\s]", doc.text)) if doc.kind == "law" else []
        if articles:
            for index, article in enumerate(articles):
                end = articles[index + 1].start() if index + 1 < len(articles) else len(doc.text)
                units.append((doc, "статья " + article.group(1), doc.text[article.start():end], article.start(), end))
            if articles[0].start() > 0:
                units.append((doc, "header", doc.text[:articles[0].start()], 0, articles[0].start()))
        elif doc.kind == "company" and boundaries:
            for index, (start, clause) in enumerate(boundaries):
                end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(doc.text)
                units.append((doc, clause, doc.text[start:end], start, end))
            if boundaries[0][0] > 0:
                units.append((doc, "header", doc.text[:boundaries[0][0]], 0, boundaries[0][0]))
        else:
            units.append((doc, "article", doc.text, 0, len(doc.text)))

    sources = []
    for doc, clause, text, offset, _ in units:
        # Prefer paragraph boundaries. A long legal paragraph is split safely
        # at whitespace, never discarded after the first fixed-length snippet.
        if len(text) <= limit:
            ranges = [(0, len(text))]
        else:
            ranges = []
            pos = 0
            while pos < len(text):
                end = min(pos + limit, len(text))
                if end < len(text):
                    newline = text.rfind("\n", pos + limit // 2, end)
                    space = text.rfind(" ", pos + limit // 2, end)
                    end = max(newline, space) + 1 if max(newline, space) >= 0 else end
                if end <= pos:
                    end = min(pos + limit, len(text))
                ranges.append((pos, end))
                pos = end
        for start, end in ranges:
            segment = text[start:end]
            excerpt = segment.strip()
            if not excerpt:
                continue
            leading = len(segment) - len(segment.lstrip())
            trailing = len(segment) - len(segment.rstrip())
            source_id = f"S{len(sources) + 1:04d}"
            sources.append(Source(
                source_id, doc.id, doc.title, doc.path, clause, excerpt,
                offset + start + leading, offset + end - trailing,
            ))
    return sources


def model_excerpt(text: str) -> str:
    """Keep the original for display but omit explicit instructions to the AI."""
    return clean_data(text)[0]


def build_links(documents: list[Document]) -> list[Link]:
    """Resolve explicit, unambiguous content references, never filename years.

    A reference is a retrieval hint, not proof of legal precedence. Merely having
    two dated editions does not establish a supersession relationship.
    """
    links, seen = [], set()
    bodies = {doc.id: clean_data(doc.text)[0] for doc in documents}
    contract_key = lambda value: normalized(value or "").replace("–", "-").replace("/", "-")
    titles = defaultdict(list)
    for doc in documents:
        title = normalized(clean_data(doc.title)[0])
        title = re.split(r"\s*\(?\s*(?:редакция|ред\.)\s+от\s+", title, maxsplit=1)[0].strip()
        title = re.split(r"\b(?:ооо|ао|пао|ип)\b", title, maxsplit=1)[0].strip()
        title = named_entity_key(title)
        # An organization name on the first line is not a document identifier.
        if len(tokens(title)) >= 2 and not re.match(r"^(?:ооо|ао|пао|ип)\b", title):
            titles[title].append(doc)

    def add(source, target, relation, basis, clause=None):
        key = (source.id, target.id, relation, clause)
        if key not in seen:
            links.append(Link(source.id, target.id, relation, basis, clause))
            seen.add(key)

    for doc in documents:
        body = bodies[doc.id]
        if "дополнительное соглашение" in body[:1500].lower() and doc.contract_number:
            matches = [target for target in documents if target.id != doc.id
                       and contract_key(target.contract_number) == contract_key(doc.contract_number)
                       and "дополнительное соглашение" not in bodies[target.id][:1500].lower()
                       and "договор" in target.title.lower()]
            if len(matches) == 1:
                add(doc, matches[0], "amends", body[:800])
        for line in body.splitlines()[1:]:
            reference = named_entity_key(line)
            for title, candidates in titles.items():
                if not re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", reference):
                    continue
                mentioned_dates = dates(line)
                if mentioned_dates:
                    candidates = [target for target in candidates if mentioned_dates & dates(
                        revision_metadata(bodies[target.id]).get("revision_date", ""))]
                if len(candidates) == 1 and candidates[0].id != doc.id:
                    clause = re.search(r"(?:\bп\.|\bпункт(?:а|е|ов)?)\s*(\d+(?:\.\d+){1,3})\b", line, re.I)
                    add(doc, candidates[0], "references", line.strip(), clause.group(1) if clause else None)
    return links


class LegalAssistant:
    def __init__(self, documents: list[Document], proxy,
                 *, use_embeddings: bool | None = None) -> None:
        self.ingestion_warnings: list[dict] = []
        self.documents = documents
        self.sources = split_sources(self.documents)
        self.by_id = {source.id: source for source in self.sources}
        self.by_document: dict[str, list[Source]] = defaultdict(list)
        for source in self.sources:
            self.by_document[source.document_id].append(source)
        self.links = build_links(self.documents)
        self.source_records = self._source_records()
        self.search_texts = ["\n".join(str(self.source_records[source.id][key] or "")
                                      for key in ("document", "contract_number", "parties", "parent_scope", "text"))
                             for source in self.sources]
        self.document_contracts = {doc.id: normalized(doc.contract_number or "").replace("–", "-").replace("/", "-")
                                   for doc in self.documents}
        self.document_entities = {doc.id: {named_entity_key(name) for name in quoted_names(clean_data(doc.text)[0])}
                                  for doc in self.documents}
        self.proxy = proxy
        self.retrieval_mode = "conditional"
        if self.retrieval_mode not in {"hybrid", "conditional"}:
            raise ValueError("Unknown RETRIEVAL_MODE")
        self.conditional_threshold = float(setting("CONDITIONAL_DENSE_THRESHOLD", default="0.5"))
        if not math.isfinite(self.conditional_threshold) or not 0 <= self.conditional_threshold <= 1:
            raise ValueError("CONDITIONAL_DENSE_THRESHOLD must be between 0 and 1")
        self.terms = [Counter(tokens(model_excerpt(source.text)) + list(search_terms(model_excerpt(source.text))))
                      for source in self.sources]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(word for terms in self.terms for word in terms)
        self.title_terms = [Counter(search_terms(" ".join(str(self.source_records[source.id][key] or "")
                                                        for key in ("document", "parent_scope", "parties", "contract_number"))))
                            for source in self.sources]
        self.title_df = Counter(word for terms in self.title_terms for word in terms)
        self.char_terms = [Counter(
            gram for word in tokens(model_excerpt(source.text)) for gram in character_trigrams(word)
        ) for source in self.sources]
        self.char_lengths = [sum(terms.values()) for terms in self.char_terms]
        self.avg_char_length = sum(self.char_lengths) / max(1, len(self.char_lengths))
        self.char_df = Counter(gram for terms in self.char_terms for gram in terms)
        self.vectors: list[list[float]] | None = None
        # Index lifecycle is owned by the v8 adapter.

    def _source_records(self) -> dict[str, dict]:
        """Citable originals and structural metadata, built once per corpus."""
        records = {}
        for doc in self.documents:
            body, _ = clean_data(doc.text)
            revision = revision_metadata(body)
            parties = " ".join(f"«{name}»" for name in quoted_names(body[:3000])[:10])
            clauses = {}
            for source in self.by_document[doc.id]:
                clauses.setdefault(source.clause, source)
            for source in self.by_document[doc.id]:
                parents = []
                if source.clause.startswith("статья "):
                    parents.append(model_excerpt(clauses[source.clause].text.splitlines()[0])[:240])
                pieces = source.clause.split(".")
                for end in range(1, len(pieces)):
                    parent = clauses.get(".".join(pieces[:end]))
                    if parent:
                        parents.append(model_excerpt(parent.text.splitlines()[0])[:240])
                records[source.id] = {"id": source.id, "document_id": doc.id,
                                      "document": model_excerpt(doc.title), "path": model_excerpt(doc.path),
                                      "clause": source.clause, "text": model_excerpt(source.text),
                                      "parent_scope": "\n".join(parents)[-600:],
                                      "contract_number": doc.contract_number, "parties": parties,
                                      "revision": revision}
        return records

    def _lexical(self, question: str) -> list[int]:
        contracts = [number.lower().replace("/", "-").replace("–", "-")
                     for number in CONTRACT_RE.findall(question)]
        entities = [named_entity_key(name) for name in quoted_names(question)]
        passage_question = CONTRACT_RE.sub(" ", question)
        # Quoted text may be either an inflected entity or a legal phrase. Keep
        # it in ordinary retrieval as well as in exact entity matching.
        query = search_terms(passage_question)
        query_grams = set().union(*(character_trigrams(word) for word in tokens(passage_question)))
        scores = []
        count = len(self.sources)
        for index, terms in enumerate(self.terms):
            score = 0.0
            for word in query:
                freq = terms.get(word, 0)
                if freq:
                    idf = math.log(1.0 + (count - self.df[word] + 0.5) / (self.df[word] + 0.5))
                    score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * self.lengths[index] / self.avg_length))
                title_freq = self.title_terms[index].get(word, 0)
                if title_freq:
                    title_idf = math.log(1.0 + (count - self.title_df[word] + 0.5) / (self.title_df[word] + 0.5))
                    score += 0.3 * title_idf * title_freq
            for gram in query_grams:
                freq = self.char_terms[index].get(gram, 0)
                if freq:
                    idf = math.log(1.0 + (count - self.char_df[gram] + 0.5) / (self.char_df[gram] + 0.5))
                    score += 0.08 * idf * freq * 2.2 / (
                        freq + 1.2 * (0.25 + 0.75 * self.char_lengths[index] / self.avg_char_length)
                    )
            doc_id = self.sources[index].document_id
            if self.document_contracts[doc_id] in contracts:
                score += 15.0
            if self.document_entities[doc_id].intersection(entities):
                score += 10.0
            scores.append(score)
        return sorted(range(count), key=lambda index: scores[index], reverse=True)

    def _dense(self, question: str) -> list[int]:
        if self.vectors is None:
            return []
        query = self.proxy.embeddings([question])[0]
        if any(len(vector) != len(query) for vector in self.vectors):
            raise ValueError("Query embedding dimension differs from index")
        scores = [sum(a * b for a, b in zip(query, vector)) for vector in self.vectors]
        return sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    def _reranker_ready(self) -> bool:
        return bool(self.proxy.rerank_model and ((self.proxy.base and self.proxy.key) or
                    (getattr(self.proxy, "rerank_base", "") and getattr(self.proxy, "rerank_key", ""))))

    def _scored_candidates(self, query: str, candidates: list[int], stage: str):
        trace = CURRENT.get()
        previous_stage = trace.stage if trace else None
        started = time.perf_counter()
        try:
            if trace:
                trace.stage = stage
            ranked = self.proxy.rerank_with_scores(query, [self.search_texts[index] for index in candidates])
            if not ranked:
                raise ValueError("Empty reranker ranking")
            order = [index for index, _ in ranked]
            chosen = [candidates[index] for index in order]
            chosen.extend(candidates[index] for index in range(len(candidates)) if index not in order)
            scores = [score for _, score in ranked]
            # Index-only, partial or raw-logit responses still provide a ranking,
            # but cannot be compared to a threshold on the 0..1 score scale.
            valid = len(ranked) == len(candidates) and all(score is not None and 0 <= score <= 1 for score in scores)
            return chosen, max(scores) if valid else None
        finally:
            if trace:
                elapsed = (time.perf_counter() - started) * 1000
                trace.timings[stage] = elapsed
                trace.timings["rerank"] = trace.timings.get("rerank", 0) + elapsed
                trace.stage = previous_stage

    def _conditional_candidates(self, query: str, lexical: list[int], candidate_k: int) -> tuple[list[int], bool]:
        chosen = lexical[:candidate_k]
        trace = CURRENT.get()
        event = {"stage": "retrieval_route", "route": "lexical", "threshold": self.conditional_threshold,
                 "rerank_score": None, "status": "reranker_unavailable"}
        dense_used = False
        try:
            if not chosen or not self._reranker_ready():
                return chosen, dense_used
            event["status"] = "lexical_rerank_failed"
            chosen, confidence = self._scored_candidates(query, chosen, "rerank_lexical")
            event["rerank_score"] = confidence
            if confidence is None:
                event["status"] = "scores_unavailable"
            elif confidence >= self.conditional_threshold:
                event["status"] = "lexical_confident"
            elif self.vectors is None:
                event["status"] = "dense_index_unavailable"
            else:
                event["status"] = "dense_search_failed"
                started = time.perf_counter()
                previous_stage = trace.stage if trace else None
                try:
                    if trace:
                        trace.stage = "dense"
                    dense = self._dense(query)[:candidate_k]
                finally:
                    if trace:
                        trace.timings["dense"] = (time.perf_counter() - started) * 1000
                        trace.stage = previous_stage
                if not dense:
                    event["status"] = "dense_empty"
                else:
                    event["status"] = "dense_rerank_failed"
                    # Keep lexical evidence until the entire alternate route succeeds.
                    alternate, dense_score = self._scored_candidates(query, dense, "rerank_dense")
                    chosen, dense_used = alternate, True
                    event.update(route="dense", status="low_lexical_score", dense_rerank_score=dense_score)
            return chosen, dense_used
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            event["type"] = type(exc).__name__
            return chosen, False
        finally:
            if trace:
                trace.events.append(event)

    def retrieve(self, question: str, limit: int | None = None) -> list[Source]:
        limit = limit or integer_setting("TOP_K", 12, 4, 30)
        candidate_k = max(limit, integer_setting("RETRIEVAL_CANDIDATES", 64, 16, 256))
        query = focused_search_question(question)
        lexical = self._lexical(query)
        if self.retrieval_mode == "conditional":
            chosen, dense_used = self._conditional_candidates(query, lexical, candidate_k)
        else:
            chosen, dense_used = self._hybrid_candidates(query, lexical, candidate_k)
        if CURRENT.get():
            CURRENT.get().events.append({"stage": "candidates", "count": len(chosen), "dense_used": dense_used})
        return self._expand_retrieval(chosen, lexical, limit)

    def _hybrid_candidates(self, query: str, lexical: list[int], candidate_k: int) -> tuple[list[int], bool]:
        try:
            dense = self._dense(query)
        except (RuntimeError, KeyError, ValueError, TypeError) as exc:
            dense = []
            if CURRENT.get():
                CURRENT.get().events.append({"stage": "dense", "status": "lexical_fallback", "type": type(exc).__name__})
        fused = defaultdict(float)
        for ranked in (lexical[:candidate_k], dense[:candidate_k]):
            for rank, index in enumerate(ranked, 1):
                fused[index] += 1.0 / (60 + rank)
        chosen = sorted(fused, key=lambda index: fused[index], reverse=True)[:candidate_k]
        # Rerank the broad candidate pool BEFORE truncating the final context.
        if self._reranker_ready():
            started = time.perf_counter()
            try:
                if CURRENT.get():
                    CURRENT.get().stage = "rerank"
                order = self.proxy.rerank(query, [self.search_texts[index] for index in chosen])
                if order:
                    chosen = [chosen[index] for index in order] + [chosen[index] for index in range(len(chosen)) if index not in order]
            except (RuntimeError, ValueError, TypeError, KeyError):
                if CURRENT.get():
                    CURRENT.get().events.append({"stage": "rerank", "status": "fused_ranking_fallback"})
            finally:
                if CURRENT.get():
                    CURRENT.get().timings["rerank"] = (time.perf_counter() - started) * 1000
        return chosen, bool(dense)

    def _expand_retrieval(self, chosen: list[int], lexical: list[int], limit: int) -> list[Source]:
        chosen = chosen[:limit]
        doc_ids = {self.sources[index].document_id for index in chosen}
        # Bring an explicitly referenced clause into context before the tail.
        prioritized = []
        for link in self.links:
            if link.source_document_id in doc_ids and link.target_clause:
                prioritized.extend(index for index, source in enumerate(self.sources)
                                   if source.document_id == link.target_document_id
                                   and source.clause == link.target_clause)
            if len(prioritized) >= 4:
                break
        chosen = list(dict.fromkeys(chosen[:8] + prioritized[:4] + chosen[8:]))
        linked_ids: set[str] = set()
        for link in self.links:
            if link.source_document_id in doc_ids or link.target_document_id in doc_ids:
                linked_ids.update((link.source_document_id, link.target_document_id))
        # Preserve related evidence even when the initial top-K is full.
        # Pick clauses using the lexical order rather than the first page.
        for doc_id in sorted(linked_ids):
            for index in (index for index in lexical if self.sources[index].document_id == doc_id):
                if index not in chosen:
                    chosen.append(index)
                if sum(self.sources[item].document_id == doc_id for item in chosen) >= 2:
                    break
        chosen = chosen[:min(limit + 8, 40)]
        return [self.sources[index] for index in chosen]



def expand_context(engine, candidates: list) -> list:
    """Bounded local clause references and continuations, without another LLM."""
    extra, seen = [], {source.id for source in candidates}
    for source in candidates[:6]:
        siblings = sorted(engine.by_document[source.document_id], key=lambda item: item.start)
        references = re.findall(r"(?:\bп\.|\bпункт[а-я]*)\s*(\d+(?:\.\d+){0,3})", source.text, re.I)
        # An explicitly resolved external reference must not resolve to an
        # unrelated same-numbered provision in this document.
        external = {link.target_clause for link in engine.links
                    if link.source_document_id == source.document_id and link.target_clause}
        additions = [other for other in siblings if other.clause in references and other.clause not in external]
        position = next(i for i, item in enumerate(siblings) if item.id == source.id)
        for index in (position - 1, position + 1):
            if 0 <= index < len(siblings) and siblings[index].clause == source.clause:
                additions.append(siblings[index])
        for other in additions:
            if other.id not in seen and len(extra) < 8:
                seen.add(other.id)
                extra.append(other)
                if CURRENT.get():
                    CURRENT.get().events.append({"stage": "structural_expansion", "from": source.id, "to": other.id})
    return candidates[:4] + extra + candidates[4:]


def pack_context(engine, candidates: list) -> list[dict]:
    """Pack complete source records. Never cut a clause halfway to fit a budget."""
    max_chars = integer_setting("CONTEXT_MAX_CHARS", 18000, 2000, 80000)
    max_sources = integer_setting("CONTEXT_MAX_SOURCES", 18, 1, 48)
    records, seen, used = [], set(), 2
    skipped = []
    for source in candidates:
        record = engine.source_records[source.id]
        # Equal text in different editions remains distinct evidence.
        key = (source.document_id, normalized(record["text"]))
        if not record["text"].strip() or key in seen:
            continue
        size = len(json.dumps(record, ensure_ascii=False)) + (2 if records else 0)
        if used + size > max_chars or len(records) >= max_sources:
            skipped.append(source.id)
            continue
        records.append(record)
        used += size
        seen.add(key)
    if CURRENT.get():
        CURRENT.get().events.append({"stage": "context", "characters": used,
                                     "source_ids": [item["id"] for item in records], "skipped_ids": skipped})
    return records
