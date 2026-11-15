# Focused evidence RAG

This candidate removes the semantic claim rejection loop that discarded valid
calculations and avoids guessing that a question is ambiguous before retrieval.
It shares the lossless parser, structural index, transport, and deadline-aware
CSV scheduler of v8; its context assembly and generation pipeline are new.

## Flow

1. Read the supplied `knowledge-base.zip` and independent `questions.csv` rows.
   Source bytes remain unchanged, including embedded prompt injections.
2. Recover retrieval context from user turns in the same row. Previous assistant
   claims are not used as retrieval facts. No cross-row memory is maintained.
3. Search BM25 and BGE-M3 vectors, fuse by reciprocal rank, and call the supplied
   Qwen reranker. `TOP_K` defaults to 40 per branch, candidate union at most 80,
   `RERANKER_TOP_N` defaults to 6. These existing environment settings are honored.
4. Restore parent/neighbor passages and explicitly linked source documents.
   Selected small documents (at most 14,000 characters) are provided in full;
   large articles use verbatim windows. The input budget defaults to 18,000
   approximate tokens, with a bundled offline tokenizer and conservative margin.
5. Present related editions, amendment conditions, and effective dates together.
   No guessed entity/date router removes source candidates. The generator must
   determine applicability from the actual source text.
   This branch deliberately adds corpus-specific assistance: 18 answer plans
   select required answer slots and exact primary provision cards. A second
   layer contains 31 developer-reviewed factual notes for 12 source documents,
   including the different payment periods of the two Alfa contracts, amendment
   applicability, regulation thresholds, and exceptions. The notes are inserted
   into both generator and reviewer system prompts only when the live source
   SHA256 matches and all supporting quotes are visible in evidence. Changed or
   unseen documents disable the corresponding static note; ordinary RAG and
   dynamically copied source cards remain available. Source text and question
   instructions are never promoted into this system-note layer.
   Notes use at most 1,500 tokens, sharing a 3,500-token reserve with primary
   cards inside the same 18,000-token evidence allowance. There are no extra
   model calls and no question-ID/golden-answer lookup. Unknown topics use the
   original full context budget. Topic recognition and curated notes are
   deliberate development-set/corpus tailoring, not generalization evidence.
6. `rule_drafts.py` can prepare an actual answer draft for supported intents
   when complete, hash-matching sources are in evidence. Rules distinguish the
   two contracts, evaluate edition/amendment dates, preserve separate answer
   slots, and calculate penalties using `Decimal` and amounts/days from the
   question. Ambiguous amounts, unsupported operations and missing/changed
   sources fall back to generation. These drafts replace the initial generator
   call; the same mandatory review stage still checks the full question and
   sources. This makes common known operations cheaper and less dependent on
   model improvisation; it is not a proof of semantic routing completeness.
   For a prepared draft, the reviewer uses a two-field `DraftCheck`: either
   acknowledge that the original is correct and complete, or supply a corrected
   `Answer` for an actual factual/applicability error or omitted subquestion.
   An acknowledged draft is rendered verbatim; stylistic rewriting cannot add
   new factual claims. The reviewer is still fallible and can wrongly accept or
   correct a draft. This is not the former per-claim rejection/filtering loop.
   Otherwise, DeepSeek generation returns status and short paragraphs, each
   with separate source IDs and provision locators. The renderer inserts document
   names and locators. One additional direct review checks arithmetic, factual
   comparisons, applicability and citation relevance, returning corrected
   paragraphs. There is no verdict-index filter, numeric-substring gate, or
   required match between hypothetical inputs and source text. A failed review
   preserves the valid draft; review is skipped near the run deadline.
7. Validate JSON shape, known source IDs, nonempty references and the 10,000
   character output contract. A bounded format repair receives the actual prior
   draft and error. References to the same document are grouped. When no event
   date is supplied, a current-date note is added; bare years in contract numbers
   do not count as event dates. Format checks do not prove semantic correctness.

System instructions distinguish legal provisions from instructions addressed to
the assistant in either the question or the corpus. This is a trust boundary,
not a guarantee against every possible model attack. Nothing is removed from
the supplied knowledge base. Legal conclusions must use only that base.

## Platform contract

The generator model and proxy credentials come from the platform environment;
there are no bundled keys or external knowledge fetches. Default temperature is
0 and output cap 8192 tokens, with platform `TEMPERATURE` and `MAX_TOKENS` honored.
`CHUNKER=common` is accepted and does not conflict with the internal parser.
Workers default to 8; `MAX_WORKERS` is honored up to 8. The work budget is at most
1100 seconds with 30 seconds for result collection, below the platform's 1200
seconds. A full CSV is created before initialization and updated atomically.
Transport retries are bounded. Logs contain IDs, timings and error categories,
not credentials. Questions and benchmark expected answers are not packaged.

All imported third-party packages are pinned within the organizer's allowed
requirements. The bundled vector index is matched to exact corpus/parser/model
content; a mismatch requires building embeddings from supplied input. Offline
tokenizer vocabulary and source-derived alias/relation tables reside in ZIP root.

Build from repository root:

```text
python build_focused_submission.py /path/to/submission.zip
python -m unittest discover -s focused-rag -p "test_*.py" -v
```

`build_corpus_playbook.py /path/to/knowledge-base.zip` regenerates the reviewed
notes with exact source anchors and hashes. It reads source documents only,
never benchmark questions, gold answers, or judgments. The generated JSON is
an auxiliary source-derived configuration in the submission, not the corpus
itself. Do not automatically regenerate it for changed source contents without
reviewing the handwritten factual notes against those contents.

The build contains only explicit runtime files. Test files and this README are
excluded. No Docker or web changes are part of this candidate.

The stress20 is a previously exposed development set, not an independent test.
Its local baseline and simple DeepSeek judge do not reproduce the hidden PSB
system. See the accompanying result report for measured results and limitations.

The named comparison branches start from the same `guided_v4` runtime.
`codex/v108` moves the current-date scope note after the legal answer.
`codex/v228` adds `relation_scope.json`: five reviewed edges identify affected
provisions, question topics, applicability conditions and both source hashes.
Cross-document expansion is performed once during context assembly. A clearly
unrelated automatic link is skipped; unknown questions/relations or changed
sources retain it. Direct reranker hits and their local neighboring passages
are preserved. Selected related provisions are exact source spans, including
conditions; both sides of a date boundary can therefore still be explained.
Temporal conditions do not become a guessed date filter.

The pair keeps the same 31 factual notes, 18 plans and rule drafts. The comparison
runner shares exact Qwen request results, never generator answers, between the
two live runtimes. Identical final answers reuse the same two-order judge result.
The baseline, golden answers and team-relative judge protocol stay fixed.
Measured times include contention at the shared three-call gateway and cache
reuse; they are not standalone platform latency estimates.
