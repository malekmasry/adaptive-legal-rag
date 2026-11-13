from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Plan(Strict):
    question: str
    domain: Literal['internal','law','both'] = 'both'
    entity: str = ''
    as_of: str = ''
    date_role: Literal['event','contract_signature','unspecified'] = 'unspecified'
    specification_date: str = ''
    subquestions: list[str] = Field(default_factory=list,max_length=5)
    rewrites: list[str] = Field(default_factory=list,max_length=2)
    clarification: str = ''
    topic: str = ''
    topic_changed: bool = False
    input_attack: bool = False
    unsafe_request: bool = False

class Citation(Strict):
    chunk_id: str
    quote: str = Field(min_length=4)

class Calculation(Strict):
    expression: str
    result: str
    unit: str
    operands: list[Citation]

class Claim(Strict):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
    calculation: Calculation | None = None

class Draft(Strict):
    status: Literal['answered','partial','not_in_base','clarification','out_of_domain']
    answer: str = ''
    claims: list[Claim] = Field(default_factory=list,max_length=20)
    conflicts: list[str] = Field(default_factory=list,max_length=8)
    warnings: list[str] = Field(default_factory=list,max_length=8)
    not_in_base: bool = False
    as_of: str = ''
    missing: list[str] = Field(default_factory=list,max_length=8)
    clarification: str = ''

class Verdict(Strict):
    supported_indices: list[int]
    issues: list[str]
    coverage: list[bool]
    supported_missing_indices: list[int] = Field(default_factory=list)

class MemorySummary(Strict):
    summary: str = Field(max_length=1800)

class Ask(Strict):
    question: str = Field(min_length=1,max_length=24000)
    session_id: str = Field(default='',max_length=128,pattern=r'^[a-zA-Z0-9_-]*$')
    mode: Literal['general','litigation'] = 'general'
