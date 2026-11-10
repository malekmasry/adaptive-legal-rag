"""Per-request accounting, isolated across concurrent CSV/chat requests."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Trace:
    started: float = field(default_factory=perf_counter)
    deadline_seconds: float = 180.0
    stage: str = "initial"
    calls: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def remaining(self) -> float:
        return max(0.0, self.deadline_seconds - (perf_counter() - self.started))

    def public(self) -> dict:
        return {
            "total_ms": round((perf_counter() - self.started) * 1000, 2),
            "timings_ms": {key: round(value, 2) for key, value in self.timings.items()},
            "api_calls": self.calls,
            "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in self.calls),
            "completion_tokens": sum(call.get("completion_tokens", 0) for call in self.calls),
            "usage_complete": all(call.get("usage_reported", False) for call in self.calls),
            "events": self.events,
        }


CURRENT: ContextVar[Trace | None] = ContextVar("rag_trace", default=None)
