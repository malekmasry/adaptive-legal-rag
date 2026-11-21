"""Strict user-authorship boundary for review premises, not a truth classifier.

Parse framing before sanitizing content. Malformed or ambiguous dialogue never
falls back to treating its entire serialization as one user-authored premise.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from rag_safety import clean_data


_ROLE_NAMES = ("пользователь", "user", "ассистент", "assistant", "система", "system",
               "разработчик", "developer", "инструмент", "tool")
_USER_ROLES = {"user", "пользователь"}
_NAME = "(?:" + "|".join(_ROLE_NAMES) + ")"
_ROLE_LINE = re.compile(r"(?im)^[ \t]*(?P<role>" + _NAME + r")[ \t]*:")
_ROLE_CUE = re.compile(r"(?i)(?<!\w)" + _NAME + r"[^\S\r\n]*(?:[:：]|->|[—–-]|=)")
_ROLE_STANDALONE = re.compile(r"(?im)^[ \t]*" + _NAME + r"[ \t]*$")
_OTHER_SPEAKER = re.compile(r"(?im)^[ \t]*(?:bot|human|ai|model|chatgpt|бот|модель|человек|помощник)\s*[:：]")
_ROLE_MARKUP = re.compile(r"(?i)<\|(?:im_start|im_end|system|developer|assistant|user|start_header_id)|"
                          r"\[/?INST\]|[<\[]/?\s*" + _NAME + r"\s*[>\]]")
_JSON_ROLE_CUE = re.compile(r'["\'](?:role|messages|content)["\']\s*:')


@dataclass(frozen=True)
class UserProvenance:
    turns: tuple[str, ...] = ()
    format: str = "ambiguous"
    ambiguous: bool = True

    @property
    def text(self) -> str:
        return "\n\n".join(self.turns)

    @property
    def latest(self) -> str:
        return self.turns[-1] if self.turns else ""


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON key")
        result[name] = value
    return result


def _nested_framing(text: str) -> bool:
    return bool(_ROLE_CUE.search(text) or _ROLE_MARKUP.search(text) or _ROLE_STANDALONE.search(text)
                or _OTHER_SPEAKER.search(text) or _JSON_ROLE_CUE.search(text))


def _result(turns: list[str], format: str) -> UserProvenance:
    # Removing instructions is separate from detecting who authored a turn.
    # Whole user texts are retained; no numeric/date/negation extraction occurs.
    cleaned = [clean_data(turn)[0].strip() for turn in turns]
    return UserProvenance(tuple(turn for turn in cleaned if turn), format, False)


def _json_turns(text: str) -> UserProvenance:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        if isinstance(value, dict) and "messages" in value:
            messages = value["messages"]
        elif isinstance(value, dict) and "role" in value:
            messages = [value]
        else:
            messages = value
        if not isinstance(messages, list) or len(messages) > 256:
            return UserProvenance()
        users = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                return UserProvenance()
            role = message["role"].strip().casefold()
            if role not in _ROLE_NAMES:
                return UserProvenance()
            if role not in _USER_ROLES:
                # Do not recursively interpret another role's content or tools.
                continue
            content = message.get("content")
            if isinstance(content, list):
                if not all(isinstance(part, dict) and part.get("type") in {"text", "input_text"}
                           and isinstance(part.get("text"), str) for part in content):
                    return UserProvenance()
                content = "\n".join(part["text"] for part in content)
            if not isinstance(content, str) or _nested_framing(content):
                return UserProvenance()
            users.append(content)
        return _result(users, "json")
    except (ValueError, TypeError, RecursionError):
        return UserProvenance()


def user_provenance(text: str) -> UserProvenance:
    """Accept plain user text, strict role lines, or typed JSON message records.

    All known non-user roles are excluded. Unknown/nested/inline/malformed role
    framing contributes no premises. This preserves negated user statements as
    written; the reviewer must still distinguish a negated value from a fact.
    """
    if not isinstance(text, str) or len(text) > 100000:
        return UserProvenance()
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        return _json_turns(stripped)
    if (_ROLE_MARKUP.search(text) or _OTHER_SPEAKER.search(text) or _JSON_ROLE_CUE.search(text)
            or _ROLE_STANDALONE.search(text)):
        return UserProvenance()
    matches = list(_ROLE_LINE.finditer(text))
    cues = list(_ROLE_CUE.finditer(text))
    if not matches:
        return UserProvenance() if cues else _result([text], "plain")
    if text[:matches[0].start()].strip():
        return UserProvenance()
    if any(not any(match.start() <= cue.start() and cue.end() <= match.end()
                   for match in matches) for cue in cues):
        return UserProvenance()
    users = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if match["role"].casefold() in _USER_ROLES:
            users.append(text[match.end():end])
    return _result(users, "role_lines")
