"""Deterministic query preparation, using only user turns and corpus names.

Adapts v8 reasoning.parse_history/fallback_plan and pipeline's explicit-entity
guard. Unlike v8's configured aliases/planner, names come from corpus text and
no model call, inferred condition, or domain-specific synonym is introduced.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from hallugraph import named_entity_key
from rag_safety import clean_data


_WORDS = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_ROLES = re.compile(
    r"(?im)^\s*(Пользователь|User|Ассистент|Assistant|System|Система|Developer)\s*:"
)
_CONTRACT = re.compile(r"(?:№\s*)?([А-ЯA-Z]{1,4}[-–]\d+[/\-]\d{4})", re.I)
_QUOTED = re.compile(r'«([^»\n]+)»|"([^"\n]+)"')
_ORG = re.compile(
    r'(?i)\b(?:ООО|АО|ПАО|ОАО|ЗАО|НКО|АНО|ГУП|МУП|ФГУП|'
    r'общество\s+с\s+ограниченной\s+ответственностью|'
    r'(?:публичное\s+)?акционерное\s+общество)\s*[«"]([^»"\n]+)[»"]'
)
_FOLLOWUP = re.compile(
    r"(?i)(?:^|[.!?]\s*)\s*(?:а\b|и\b|тогда\b|значит\b|теперь\b|сколько\b|"
    r"како[йев]\s+срок|как\s+(?:им|ему|ей|их)\b)|"
    r"\b(?:по\s+нему|по\s+ней|по\s+ним|у\s+них|им|его|ее|её|их|"
    r"этот|эта|это|эти|этому|этого|этой|наш\w*|того|такой)\b"
)
_TOPIC_RESET = re.compile(
    r"(?i)\b(?:другой\s+вопрос|новая\s+тема|сменим\s+тему|"
    r"теперь\s+(?:о|про)|не\s+связан\w*\s+с)\b"
)
_CONFUSABLES = str.maketrans("aceopxykmtbh", "асеорхукмтвн")


def user_turns(text: str) -> list[str]:
    """Read serialized dialogue as data; assistant/system content is discarded."""
    if text.lstrip().startswith(("[", "{")):
        try:
            data = json.loads(text)
            messages = data.get("messages") if isinstance(data, dict) else data
            if isinstance(messages, list):
                users = [m["content"].strip() for m in messages
                         if isinstance(m, dict) and m.get("role") == "user"
                         and isinstance(m.get("content"), str) and m["content"].strip()]
                if users:
                    return users
        except (ValueError, TypeError):
            pass
    matches = list(_ROLES.finditer(text))
    users = []
    for index, match in enumerate(matches):
        if match[1].lower() in {"пользователь", "user"}:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.end():end].strip()
            if content:
                users.append(content)
    return users or [text]


def latest_user_turn(text: str) -> str:
    return user_turns(text)[-1]


def _surface(word: str) -> str:
    word = word.casefold().replace("ё", "е")
    # A Latin organization stays Latin. Repair only mixed-script word tokens.
    if re.search(r"[а-я]", word):
        word = word.translate(_CONFUSABLES)
    return word


def _words(text: str) -> tuple[str, ...]:
    return tuple(_surface(word) for word in _WORDS.findall(text))


def _keys(words: tuple[str, ...]) -> tuple[str, ...]:
    # The shared evidence key omits these adjective accusative/instrumental
    # endings. Extend only retrieval matching; evidence validation is unchanged.
    return tuple(word[:-2] if len(word) >= 6 and word.endswith(("ую", "юю", "ою", "ею"))
                 else named_entity_key(word) for word in words)


def _one_edit(a: str, b: str) -> bool:
    """One insertion/deletion/substitution/transposition, never a prefix match."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        different = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(different) == 1 or (
            len(different) == 2 and different[1] == different[0] + 1
            and a[different[0]] == b[different[1]] and a[different[1]] == b[different[0]]
        )
    shorter, longer = sorted((a, b), key=len)
    for i, (x, y) in enumerate(zip(shorter, longer)):
        if x != y:
            return shorter[i:] == longer[i + 1:]
    return True


class QueryBridge:
    def __init__(self, documents=()):
        names = set()
        for doc in documents:
            body, _ = clean_data(doc.text)
            # Organization forms distinguish party names from quoted clauses.
            for match in _ORG.finditer(body):
                name = match[1].strip()
                words = _words(name)
                if 1 <= len(words) <= 7 and 4 <= len(name) <= 100:
                    names.add(name)
        canonical = {}
        for name in sorted(names):
            canonical.setdefault(_words(name), name)
        self.names = tuple(canonical.values())
        self.surface = defaultdict(set)
        self.stems = defaultdict(set)
        for name in self.names:
            words = _words(name)
            self.surface[words].add(name)
            self.stems[_keys(words)].add(name)
        self.lengths = sorted({len(words) for words in self.surface}, reverse=True)

    def resolve(self, text: str) -> tuple[str, ...]:
        """Resolve complete name spans. Ambiguity yields no added canonical name."""
        words = _words(text)
        keys = _keys(words)
        found = set()
        occupied = set()
        pending = []
        for size in self.lengths:
            for start in range(len(words) - size + 1):
                span = set(range(start, start + size))
                if occupied.intersection(span):
                    continue
                surface = words[start:start + size]
                exact = self.surface.get(surface, set())
                candidates = exact or self.stems.get(keys[start:start + size], set())
                if candidates:
                    occupied.update(span)
                    if len(candidates) == 1:
                        found.update(candidates)
                else:
                    pending.append((start, size))
        # Typos are allowed only in long words, with all other name tokens exact.
        # Every equally close corpus name is considered, so confusable companies
        # cannot be resolved by the ordering of the documents.
        for start, size in pending:
            span = set(range(start, start + size))
            if occupied.intersection(span):
                continue
            candidates = set()
            # Surface comparison also handles typos in a noun ending that a
            # heuristic stemmer could mistake for an adjective inflection.
            for key, catalog in ((words[start:start + size], self.surface),
                                 (keys[start:start + size], self.stems)):
                for known, names in catalog.items():
                    if len(known) != size:
                        continue
                    differences = [(a, b) for a, b in zip(key, known) if a != b]
                    if len(differences) == 1:
                        a, b = differences[0]
                        if min(len(a), len(b)) >= 6 and _one_edit(a, b):
                            candidates.update(names)
            if len(candidates) == 1:
                found.update(candidates)
                occupied.update(span)
        return tuple(sorted(found))

    def _anchors(self, text: str) -> list[str]:
        contracts = _CONTRACT.findall(text)
        names = list(self.resolve(text))
        # Without a corpus (legacy public helper), preserve explicit quoted data.
        if not self.names:
            names = [a or b for a, b in _QUOTED.findall(text)]
        return list(dict.fromkeys(contracts + [f"«{name}»" for name in names]))

    def prepare(self, text: str) -> str:
        turns = user_turns(text)
        latest = turns[-1]
        names = self.resolve(latest)
        explicit = bool(names or _CONTRACT.search(latest) or _QUOTED.search(latest))
        query = latest
        if not explicit and len(turns) > 1 and _FOLLOWUP.search(latest) and not _TOPIC_RESET.search(latest):
            inherited = False
            for earlier in reversed(turns[:-1][-3:]):
                anchors = self._anchors(earlier)
                if anchors:
                    query += "\n" + " ".join(anchors)
                    inherited = True
                    break
                if _TOPIC_RESET.search(earlier):
                    break
            if not inherited and not _TOPIC_RESET.search(turns[-2]):
                # A follow-up about a legal topic may have no company at all.
                # Preserve one bounded user turn, never an assistant's answer.
                query += "\nКонтекст пользователя: " + turns[-2][-600:]
        # All additions are names/identifiers observed in the corpus or user
        # turns. The current question, including negation/numbers, stays intact.
        for name in names:
            quoted = f"«{name}»"
            if quoted not in query:
                query += " " + quoted
        return query


def focused_query(text: str) -> str:
    # The public helper has no corpus. Preserve user-turn boundaries until the
    # assistant's corpus-aware prepare() can decide which entity is current.
    turns = user_turns(text)
    return text if len(turns) == 1 else "\n".join(f"Пользователь: {turn}" for turn in turns[-4:])
