"""Request-local dialogue rules and explicit, isolated in-memory chat sessions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from threading import RLock

from rag_safety import clean_data, normalized


ROLES = re.compile(r"(?im)^[ \t]*(Пользователь|Ассистент|User|Assistant):[ \t]*")
PREVIOUS = r"(?:предыдущий|предидущий|прошлый|последний)"
REPEAT = re.compile(
    rf"(?:повтори(?:те)?|повторить|покажи(?:те)?|пришли(?:те)?) "
    rf"(?:(?:свой|твой|ваш) )?(?:{PREVIOUS} )?ответ(?: (?:еще раз|снова|дословно))?"
    r"|(?:повтори(?:те)?|повторить)(?: (?:еще раз|снова|сказанное))?")
REANSWER = re.compile(rf"ответь(?:те)? (?:еще раз )?на (?:(?:мой|тот) )?{PREVIOUS} вопрос(?: еще раз)?"
                     r"|ответь(?:те)? на (?:него|вопрос выше)")


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class Request:
    conversation: str
    latest_question: str
    search_question: str
    route: str = "search"
    repeated_text: str = ""
    history_turns: int = 0


def parse_turns(text: str) -> list[Turn]:
    matches = list(ROLES.finditer(text))
    if not matches:
        return [Turn("user", text)]
    result = []
    if text[:matches[0].start()].strip():
        result.append(Turn("user", text[:matches[0].start()].strip()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        role = "user" if match[1].casefold() in {"пользователь", "user"} else "assistant"
        result.append(Turn(role, text[match.end():end].strip()))
    return result


def intent(text: str) -> str:
    value = normalized(text)
    value = re.sub(r"\bпожалуйста\b", "", value)
    value = re.sub(r"[.,!?;:]+", " ", value)
    value = " ".join(value.split())
    value = re.sub(r"^(?:можешь|можете|прошу|не мог бы ты|не могли бы вы) ", "", value)
    if REPEAT.fullmatch(value):
        return "repeat"
    if REANSWER.fullmatch(value):
        return "reanswer"
    return "question"


def identifiers(text):
    from legal_core import CONTRACT_RE, quoted_names
    return CONTRACT_RE.findall(text) + [f"«{name}»" for name in quoted_names(text)]


def without_identifiers(text):
    from legal_core import CONTRACT_RE
    value = CONTRACT_RE.sub("", text)
    return re.sub(r'«[^»]+»|"[^"\n]+"', "", value).strip()


def elliptical(text):
    value = normalized(text).strip(" .?!")
    filler = {"а", "у", "по", "для", "с", "теперь", "этот", "тот", "да", "нет", "же",
              "договор", "договору", "договором", "ооо", "ао", "пао", "ип", "пожалуйста"}
    words = re.findall(r"\w+", normalized(without_identifiers(text)))
    entity_only = bool(identifiers(text)) and all(word in filler for word in words)
    short = value in {"а сколько", "а когда", "а почему", "а какой срок", "а срок", "почему",
                      "поясни", "поясните", "подробнее", "расскажи подробнее", "да", "нет"}
    return entity_only, short


def resolve(question: str, *, turns: list[Turn] | None = None) -> Request:
    """Resolve only user context; assistant assertions never seed retrieval."""
    if len(question) > 20000:
        # Preserve the existing pipeline's size rejection without first parsing
        # an unbounded transcript or copying an oversized historical answer.
        return Request(question, question, question)
    turns = parse_turns(question) if turns is None else turns
    if not turns or turns[-1].role != "user":
        return Request(question, "", "", "missing_question", history_turns=len(turns))
    latest = clean_data(turns[-1].text)[0].strip()
    if not latest:
        return Request(question, "", "", "missing_question", history_turns=max(0, len(turns)-1))
    previous_search = ""
    active_identifiers = []
    resolved_latest = latest
    route = "search"
    current_search = ""
    for turn in turns:
        if turn.role != "user":
            continue
        current = clean_data(turn.text)[0].strip()
        if not current:
            continue
        kind = intent(current)
        if kind == "repeat":
            route = "repeat"
            continue
        if kind == "reanswer":
            route = "reanswer" if previous_search else "missing_history"
            current_search = previous_search
            resolved_latest = previous_search
            continue
        own_identifiers = identifiers(current)
        if re.match(r"(?i)^(?:(?:новый|другой) вопрос|новая тема|теперь о другом)\s*[:.—-]", current):
            previous_search, active_identifiers = "", []
        entity_only, short = elliptical(current)
        if (entity_only or short) and previous_search:
            topic = without_identifiers(previous_search) if own_identifiers else previous_search
            current_search = f"{topic}\nУточнение пользователя: {current}"
            resolved_latest = current_search
            route = "contextual_followup"
        elif (entity_only or short) and not previous_search:
            route, current_search, resolved_latest = "missing_history", "", current
        else:
            current_search = current
            if not own_identifiers and active_identifiers:
                current_search += " " + " ".join(active_identifiers)
            elif not own_identifiers and previous_search:
                # Preserve the baseline's user-only context when there is no
                # named contract/entity to carry forward (e.g. handing over
                # an original document followed by "Да. В юротдел ...?").
                current_search = previous_search + "\n" + current
            resolved_latest, route = current, "search"
        if own_identifiers:
            active_identifiers = own_identifiers
        if current_search:
            previous_search = current_search
    if intent(latest) == "repeat":
        previous = next((turn.text for turn in reversed(turns[:-1]) if turn.role == "assistant"), "")
        return Request(question, latest, "", "repeat" if previous else "missing_answer",
                       previous, len(turns)-1)
    return Request(question, resolved_latest, current_search, route, history_turns=len(turns)-1)


def immediate_answer(request: Request):
    """Zero-search responses; imported history is explicitly unverified text."""
    from legal_core import Answer
    from single_pass import irac_notice
    from rag_telemetry import Trace
    if request.route not in {"repeat", "missing_history", "missing_answer", "missing_question"}:
        return None
    trace = Trace()
    if request.route == "repeat":
        text, _ = clean_data(request.repeated_text)
        # User-supplied source IDs do not turn a pasted answer into verified evidence.
        text = re.sub(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", "", text).strip()
        if text:
            output = "Предыдущая реплика ассистента из переданной истории (без проверки по базе):\n\n" + text
            status = "repeated_history"
            if len(output) > 10000:
                output = irac_notice("Повтор предыдущего ответа.",
                                     "Предыдущий ответ слишком длинный для повтора целиком. Уточните, какую часть повторить.")
                status = "clarification"
        else:
            output = irac_notice("Повтор предыдущего ответа.", "Нет доступного текста для повторения. Повторите сам вопрос.")
            status = "clarification"
    else:
        reason = {"missing_history": "Предыдущего вопроса в доступной истории нет. Напишите его ещё раз.",
                  "missing_answer": "Предыдущего ответа в доступной истории нет. Напишите вопрос, на который нужно ответить.",
                  "missing_question": "Уточните юридический вопрос по документам."}[request.route]
        output, status = irac_notice("Уточнение диалога.", reason), "clarification"
    trace.events.append({"stage": "dialogue", "route": request.route, "history_turns": request.history_turns})
    diagnostics = trace.public()
    diagnostics.update(pipeline="dialogue", pipeline_calls=0, retrieval_rounds=0, search_total_ms=0,
                       semantic_verification=False)
    return Answer(output, [], status, diagnostics)


class ChatSession:
    """One conversation, one lock. No disk persistence or shared user history.

    A fresh session must be created for each chat. CSV rows intentionally use
    engine.answer directly, with their own supplied transcript instead.
    """
    def __init__(self, engine, *, max_exchanges: int = 6, max_chars: int = 16000):
        if not 1 <= max_exchanges <= 50 or not 1000 <= max_chars <= 18000:
            raise ValueError("Invalid session memory limits")
        self.engine = engine
        self.max_exchanges, self.max_chars = max_exchanges, max_chars
        self._turns: list[Turn] = []
        self._last_answer = None
        self._lock = RLock()

    @property
    def history(self):
        with self._lock:
            return tuple(self._turns)

    def clear(self):
        with self._lock:
            self._turns.clear()
            self._last_answer = None

    def ask(self, question: str):
        from legal_core import Answer
        from rag_telemetry import Trace
        with self._lock:
            if intent(question) == "repeat" and self._last_answer is not None:
                cached = self._last_answer
                trace = Trace()
                trace.events.append({"stage": "dialogue", "route": "repeat_cached",
                                     "history_turns": len(self._turns)})
                diagnostics = trace.public()
                diagnostics.update(pipeline="dialogue", pipeline_calls=0, retrieval_rounds=0,
                                   search_total_ms=0, semantic_verification=False)
                answer = Answer(cached.text, list(cached.sources), cached.status, diagnostics)
            else:
                # Evict whole exchanges, never silently cut legal conditions.
                while self._turns and sum(len(t.text)+15 for t in self._turns) + len(question)+15 > 20000:
                    del self._turns[:2]
                turns = self._turns + [Turn("user", question)]
                conversation = "\n".join(("Пользователь: " if t.role == "user" else "Ассистент: ") + t.text
                                         for t in turns)
                request = resolve(conversation, turns=turns)
                answer = self.engine._answer_dialogue(request)
            self._turns.extend([Turn("user", question), Turn("assistant", answer.text)])
            while self._turns and (len(self._turns) > 2*self.max_exchanges or
                                  sum(len(t.text)+15 for t in self._turns) > self.max_chars):
                del self._turns[:2]
            self._last_answer = deepcopy(answer) if self._turns else None
            return answer
