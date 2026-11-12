"""HalluGraph-inspired auditable alignment for Russian legal evidence.

Implements entity grounding and directed relation preservation, with source-local
provenance. This is an adaptation, not the paper's spaCy/SLM implementation.
Graph extraction and semantic entailment remain fallible model judgments.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal

from rag_safety import normalized, safe_display, string, strings


MONTHS = {name: index for index, name in enumerate(
    ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
     "сентября", "октября", "ноября", "декабря"), 1)}
DATE_PATTERN = r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b"
CONTRACT_PATTERN = r"[А-ЯA-Z]{1,4}[-–]\d+[/\-]\d{4}"


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


def applicable(event: date, boundary: date, operator: str) -> bool:
    operations = {"after": event > boundary, "from": event >= boundary,
                  "before": event < boundary, "through": event <= boundary}
    if operator not in operations:
        raise ValueError("Unsupported date operator")
    return operations[operator]


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


@dataclass(frozen=True)
class Fact:
    id: str
    source_id: str
    quote: str
    subject: str
    relation: str
    object: str
    conditions: str

    def public(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Claim:
    id: str
    section: str
    text: str
    fact_ids: tuple[str, ...]
    subject: str
    relation: str
    object: str
    conditions: str

    def public(self) -> dict:
        return asdict(self)


def read_facts(items, evidence: dict[str, dict]) -> list[Fact]:
    if not isinstance(items, list) or len(items) > 40:
        raise ValueError("Invalid source graph")
    facts = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid fact")
        source_id = string(item.get("source_id"), 16)
        quote = string(item.get("quote"), 1800)
        # Never trust model-created source IDs or quotations. Do not permit quotes
        # in metadata to substantiate obligations in the document body.
        if source_id not in evidence or len(quote) < 8:
            continue
        if normalized(quote) not in normalized(evidence[source_id]["text"]):
            continue
        values = [string(item.get(key), 500) for key in ("subject", "relation", "object", "conditions")]
        if not all(values[:3]) or not all(safe_display(value) for value in values + [quote]):
            continue
        facts.append(Fact(f"F{len(facts) + 1}", source_id, quote, *values))
    return facts


def read_claims(items, facts: list[Fact]) -> list[Claim]:
    if not isinstance(items, list) or len(items) > 12:
        raise ValueError("Invalid answer graph")
    allowed = {fact.id for fact in facts}
    claims = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid claim")
        text = string(item.get("text"), 1200)
        section = string(item.get("section"), 16)
        if section not in {"rule", "application", "conclusion"}:
            raise ValueError("Invalid IRAC section")
        refs = tuple(dict.fromkeys(strings(item.get("fact_ids"), 8, 16)))
        values = [string(item.get(key), 500) for key in ("subject", "relation", "object", "conditions")]
        if not text or not refs or not set(refs) <= allowed or not all(values[:3]):
            raise ValueError("Ungrounded claim")
        if not all(safe_display(value) for value in values + [text]) or re.search(r"\[S\d+", text):
            raise ValueError("Unsafe claim or model-written citation")
        claims.append(Claim(f"C{len(claims) + 1}", section, text, refs, *values))
    return claims


def numeric_atoms(text: str) -> set[str]:
    # Preserve decimal values, ignore source IDs. Calculations are not authorized
    # merely by copying their inputs; unsupported results require correction.
    text = normalized(text)
    partial_dates = set()
    def partial_date(match):
        partial_dates.add(f"date:{MONTHS[match[2]]:02d}{int(match[1]):02d}")
        return " "
    text = re.sub(r"\b[SFСC]\d+\b", "", text, flags=re.I)
    text = re.sub(CONTRACT_PATTERN, "", text, flags=re.I)
    # Whole-hour clock notation and its verbal hour are the same circumstance.
    text = re.sub(r"\b(\d{1,2}):00\b", lambda m: str(int(m[1])), text)
    def word_date(match):
        return f"{int(match.group(3)):04d}{MONTHS[match.group(2).lower()]:02d}{int(match.group(1)):02d}"
    text = re.sub(DATE_PATTERN, word_date, text, flags=re.I)
    text = re.sub(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", lambda m: m[3] + m[2] + m[1], text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: m[1] + m[2] + m[3], text)
    text = re.sub(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\b", partial_date, text)
    number_words = {"один": "1", "одного": "1", "одной": "1", "одна": "1", "одну": "1",
                    "два": "2", "двух": "2", "две": "2", "три": "3", "трех": "3",
                    "четыре": "4", "четырех": "4", "пять": "5", "пяти": "5", "шесть": "6", "шести": "6",
                    "семь": "7", "семи": "7", "восемь": "8", "восьми": "8", "девять": "9", "девяти": "9",
                    "десять": "10", "десяти": "10", "одиннадцать": "11", "одиннадцати": "11",
                    "двенадцать": "12", "двенадцати": "12", "тринадцать": "13", "тринадцати": "13",
                    "четырнадцать": "14", "четырнадцати": "14", "пятнадцать": "15", "пятнадцати": "15",
                    "шестнадцать": "16", "шестнадцати": "16", "семнадцать": "17", "семнадцати": "17",
                    "восемнадцать": "18", "восемнадцати": "18", "девятнадцать": "19", "девятнадцати": "19"}
    tens = {"двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30,
            "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесяти": 50,
            "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
            "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90}
    text = text.replace("ё", "е")
    text = re.sub(r"\b(" + "|".join(tens) + r")(?:\s+(" + "|".join(number_words) + r"))?\b",
                  lambda m: str(tens[m[1].lower()] + int(number_words[m[2].lower()]) if m[2] else tens[m[1].lower()]), text, flags=re.I)
    text = re.sub(r"\b(" + "|".join(number_words) + r")\b(?=\s+(?:рабоч|календар|дн|день|дня|месяц|час|лет|год|раз|миллион|тысяч|рубл|процент))",
                  lambda m: number_words[m[0].lower()], text, flags=re.I)
    def scale(match):
        multiplier = 1_000_000 if match.group(2).lower().startswith(("млн", "миллион")) else 1000
        return str(Decimal(match.group(1).replace(",", ".")) * multiplier)
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*(млн\.?|миллион[а-я]*|тыс\.?|тысяч[а-я]*)", scale, text, flags=re.I)
    values = partial_dates
    for value in re.findall(r"(?<!\w)\d+(?: \d{3})*(?:[.,]\d+)?(?!\w)", text):
        integer, dot, fraction = value.replace(" ", "").replace(",", ".").partition(".")
        integer = integer.lstrip("0") or "0"
        fraction = fraction.rstrip("0")
        values.add(integer + ("." + fraction if dot and fraction else ""))
    return values


def align(facts: list[Fact], claims: list[Claim], evidence: dict | None = None, question: str = "") -> dict:
    by_id = {fact.id: fact for fact in facts}
    entity_total = entity_good = relation_good = 0
    findings, edges = [], []
    for claim in claims:
        support = [by_id[ref] for ref in claim.fact_ids]
        nodes = {normalized(value) for fact in support for value in (fact.subject, fact.object)}
        endpoints = (normalized(claim.subject), normalized(claim.object))
        entity_total += 2
        entity_good += sum(node in nodes for node in endpoints)
        matched = [fact.id for fact in support if
                   (normalized(fact.subject), normalized(fact.object)) == endpoints and
                   normalized(fact.relation) == normalized(claim.relation)]
        relation_good += bool(matched)
        issues = []
        if not all(node in nodes for node in endpoints):
            issues.append("entity_substitution")
        if not matched:
            issues.append("unsupported_relation")
        if any((normalized(fact.object), normalized(fact.subject)) == endpoints and
               normalized(fact.relation) == normalized(claim.relation) for fact in support):
            issues.append("reversed_relation")
        if claim.section == "rule" and any(
                (normalized(fact.subject), normalized(fact.object)) == endpoints and
                normalized(claim.relation).startswith("не ") and
                normalized(claim.relation)[3:] == normalized(fact.relation) for fact in support):
            issues.append("relation_polarity_changed")
        quote_text = " ".join(fact.quote for fact in support)
        allowed_numbers = numeric_atoms(quote_text)
        date_evidence = quote_text
        if claim.section in {"application", "conclusion"}:
            allowed_numbers.update(numeric_atoms(question))
            date_evidence += " " + question
        if evidence:
            date_evidence += " " + " ".join(evidence.get(fact.source_id, {}).get("text", "") for fact in support)
            date_evidence += " " + " ".join(str(value) for fact in support for value in
                evidence.get(fact.source_id, {}).get("revision", {}).values())
            allowed_numbers.update(numeric_atoms("; ".join(
                evidence.get(fact.source_id, {}).get("text", "") for fact in support)))
            allowed_numbers.update(numeric_atoms(" ".join(
                str(value) for fact in support for value in
                evidence.get(fact.source_id, {}).get("revision", {}).values())))
            for fact in support:
                for value in evidence.get(fact.source_id, {}).get("revision", {}).values():
                    allowed_numbers.update(str(item.year) for item in dates(value))
        # User-supplied event dates are circumstances, not legal deadlines.
        for event_date in dates(question):
            allowed_numbers.add(event_date.strftime("%Y%m%d"))
        # A partial calendar date is checked as month/day, never as an amount.
        allowed_numbers.update(f"date:{item.month:02d}{item.day:02d}" for item in dates(date_evidence))
        if not numeric_atoms(claim.text) <= allowed_numbers:
            issues.append("unsupported_number")
        if evidence is not None:
            metadata = " ".join(str(evidence.get(fact.source_id, {}).get(key, ""))
                                for fact in support for key in ("contract_number", "document", "text", "parties"))
            contract_key = lambda value: normalized(value).replace("–", "-").replace("/", "-")
            known = {contract_key(value) for value in re.findall(CONTRACT_PATTERN, metadata, re.I)}
            if claim.section in {"application", "conclusion"}:
                known.update(contract_key(value) for value in re.findall(CONTRACT_PATTERN, question, re.I))
            used = {contract_key(value) for value in re.findall(CONTRACT_PATTERN, claim.text, re.I)}
            if not used <= known:
                issues.append("unsupported_contract")
            named = {named_entity_key(name) for name in re.findall(r"«([^»]+)»", claim.text)
                     if name and name[0].isupper()}
            grounded = {named_entity_key(name) for name in re.findall(r"«([^»]+)»", metadata + " " + quote_text)}
            if not named <= grounded:
                issues.append("unsupported_named_entity")
            inclusive_spans = re.findall(r"(?:\bдо|\bпо)\s+(.{5,55}?)\s+(?:года\s+|г\.\s+)?включительно", claim.text, re.I)
            inclusive_dates = set().union(*(dates(span) for span in inclusive_spans))
            excluded_dates = set().union(*(dates(evidence.get(fact.source_id, {}).get("revision", {}).get(
                "effective_until_exclusive", "")) for fact in support))
            if inclusive_dates & excluded_dates:
                issues.append("inclusive_exclusive_conflict")
        # A source's conditional edge must not become unconditional in the graph.
        if any(fact.conditions for fact in support) and not claim.conditions:
            issues.append("dropped_condition")
        if issues:
            findings.append({"claim_id": claim.id, "issues": issues})
        edges.append({"claim_id": claim.id, "source_fact_ids": matched,
                      "source_ids": list(dict.fromkeys(fact.source_id for fact in support))})
    return {"entity_grounding": entity_good / entity_total if entity_total else None,
            "relation_preservation": relation_good / len(claims) if claims else None,
            "findings": findings, "alignments": edges,
            "source_graph": [fact.public() for fact in facts],
            "answer_graph": [claim.public() for claim in claims]}


def check_temporal(items, question: str, evidence: dict[str, dict]) -> list[str]:
    """Verify explicit comparisons, without guessing which event a date describes."""
    if not isinstance(items, list) or len(items) > 12:
        raise ValueError("Invalid temporal checks")
    issues = []
    markers = {"after": r"\bпосле\b", "from": r"\bс\b", "before": r"\bдо\b",
               "through": r"включительно|не позднее"}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid temporal check")
        try:
            event = date.fromisoformat(string(item.get("event_date"), 10))
            boundary = date.fromisoformat(string(item.get("boundary"), 10))
        except ValueError:
            issues.append("unverifiable_date_comparison")
            continue
        source_id = string(item.get("source_id"), 16)
        quote = string(item.get("quote"), 1800)
        operator = string(item.get("operator"), 16)
        revision = evidence.get(source_id, {}).get("revision", {})
        field = {"from": "effective_from", "before": "effective_until_exclusive"}.get(operator)
        metadata_proof = bool(field and boundary in dates(revision.get(field, "")))
        quoted_proof = bool(operator in markers and boundary in dates(quote) and re.search(markers[operator], quote, re.I))
        question_dates = dates(question)
        event_grounded = event in question_dates or (not question_dates and event == boundary)
        if (source_id not in evidence or normalized(quote) not in normalized(evidence[source_id]["text"])
                or not event_grounded or not (quoted_proof or metadata_proof)
                or type(item.get("applies")) is not bool):
            issues.append("unverifiable_date_comparison")
            continue
        if applicable(event, boundary, operator) != item["applies"]:
            issues.append("wrong_date_boundary")
    return issues
