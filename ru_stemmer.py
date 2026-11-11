"""Russian Snowball algorithm implemented with Python's standard library.

Specification: https://snowballstem.org/algorithms/russian/stemmer.html
Only lexical index/query tokens use these stems; source text is preserved.
"""
from functools import lru_cache

VOWELS = frozenset("аеиоуыэюя")


def _suffixes(unconditional: str, conditional: str = "") -> tuple[tuple[str, bool], ...]:
    rules = [(suffix, False) for suffix in unconditional.split()]
    rules.extend((suffix, True) for suffix in conditional.split())
    return tuple(sorted(rules, key=lambda item: len(item[0]), reverse=True))


GERUND = _suffixes("ив ивши ившись ыв ывши ывшись", "в вши вшись")
REFLEXIVE = _suffixes("ся сь")
ADJECTIVE = _suffixes("ее ие ые ое ими ыми ей ий ый ой ем им ым ом его ого ему ому их ых ую юю ая яя ою ею")
PARTICIPLE = _suffixes("ивш ывш ующ", "ем нн вш ющ щ")
VERB = _suffixes(
    "ила ыла ена ейте уйте ите или ыли ей уй ил ыл им ым ен ило ыло ено ят ует уют ит ыт ены ить ыть ишь ую ю",
    "ла на ете йте ли й л ем н ло но ет ют ны ть ешь нно",
)
NOUN = _suffixes("а ев ов ие ье е иями ями ами еи ии и ией ей ой ий й иям ям ием ем ам ом о у ах иях ях ы ь ию ью ю ия ья я")
DERIVATIONAL = _suffixes("ост ость")
SUPERLATIVE = _suffixes("ейш ейше")


def _region(word: str, start: int) -> int:
    """Position after the first vowel/non-vowel pair within a region."""
    for index in range(start + 1, len(word)):
        if word[index - 1] in VOWELS and word[index] not in VOWELS:
            return index + 1
    return len(word)


def _remove(word: str, region: int, rules: tuple[tuple[str, bool], ...]) -> tuple[str, bool]:
    for suffix, needs_vowel in rules:
        start = len(word) - len(suffix)
        if start < region or not word.endswith(suffix):
            continue
        # Select the longest ending first. A failed condition does not permit
        # falling through to a shorter ending from the same suffix group.
        if needs_vowel and (start <= region or word[start - 1] not in "ая"):
            return word, False
        return word[:start], True
    return word, False


@lru_cache(maxsize=65536)
def stem_russian(word: str) -> str:
    """Normalize one token; retain numbers, Latin and mixed identifiers."""
    word = word.lower().replace("ё", "е")
    if not word or not all("а" <= character <= "я" for character in word):
        return word
    rv = next((i + 1 for i, character in enumerate(word) if character in VOWELS), len(word))
    r2 = _region(word, _region(word, 0))
    word, removed = _remove(word, rv, GERUND)
    if not removed:
        word, _ = _remove(word, rv, REFLEXIVE)
        word, adjective = _remove(word, rv, ADJECTIVE)
        if adjective:
            word, _ = _remove(word, rv, PARTICIPLE)
        else:
            word, verb = _remove(word, rv, VERB)
            if not verb:
                word, _ = _remove(word, rv, NOUN)
    if len(word) > rv and word.endswith("и"):
        word = word[:-1]
    word, _ = _remove(word, max(rv, r2), DERIVATIONAL)
    if len(word) - 2 >= rv and word.endswith("нн"):
        word = word[:-1]
    else:
        word, superlative = _remove(word, rv, SUPERLATIVE)
        if superlative:
            if len(word) - 2 >= rv and word.endswith("нн"):
                word = word[:-1]
        elif len(word) > rv and word.endswith("ь"):
            word = word[:-1]
    return word
