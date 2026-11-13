# Retriever from conditional35 + v8 answer pipeline

This branch combines the retrieval implementation at `5d7f10d193edfca14f8c94a7f343e4f70303cc6a`
(`dev_if_test`, user-reported platform score approximately 35) with the answer
pipeline at `a61d1e89c096cea69902d2e6e646bee3bcda5496` (`less-secure`, v8,
user-reported score 12.5 and platform elapsed time 690 seconds).
Those historical scores are not measurements of this combination.

The submission runtime lives in `psb-rag/`. Root-level legacy entry points are
historical and are not packaged. Build using `psb-rag/build_submission.py`.

## Boundary

`retriever35_core.py` contains only donor splitting, lexical ranking, conditional
dense routing, relation expansion and context selection. It has no donor answer
generator, IRAC formatter, claim checker or independent HTTP client.
`retriever35.py` adapts its output to the v8 evidence contract and uses the v8 API
transport. `pipeline.py` imports this adapter instead of the old retriever.

The generator prompt, verifier, correction policy, planner, conversation memory,
claim validation, injection defenses, output schema, CSV runner, retry policy,
1100-second working deadline and finalization behavior remain from v8.
The original `retrieval.py` remains for v8 entity/date helpers; its `Retriever`
class is not used by the submission pipeline.

## Retrieval

1. Split the original documents using the donor algorithm: clauses/articles,
   paragraph/whitespace boundaries, up to 1400 characters per source fragment.
   The supplied 21-document KB produces 342 fragments. Raw source text remains
   intact. Offsets refer to the original document.
2. Use the v8 planner's standalone question, then the donor's question cleanup.
3. Rank using the donor term, title, character-trigram, contract-number and
   quoted-entity scoring. Take 64 candidates by default.
4. Send the question and all candidates to Qwen3-Reranker-8B. Preserve the donor's
   unprefixed query and complete scored ranking. A maximum valid relevance score
   of at least 0.5 keeps the lexical ranking.
5. If the score is below 0.5, rank the entire source corpus by BGE-M3 dot product,
   take 64 dense candidates, and rerank that pool. A successful dense route
   REPLACES the lexical ranking; it does not fuse the two rankings.
6. Take the first 12 sources, add explicitly linked clauses/related documents,
   then referenced clauses and adjacent continuations. Deduplicate and pack
   whole source records into the donor's 18,000-character / 18-source limit.
7. Convert selected sources to v8 evidence with original text, citation IDs,
   metadata, editorial notes and instruction annotations. The unchanged v8
   context builder additionally enforces its 18,000-token budget. Donor packing
   measures its sanitized search records; v8 receives raw evidence, so those
   two budgets intentionally measure different representations.

No v8 MMR, RRF, additional reranker pass or date/entity pool filter is applied
after donor selection. The v8 pre-retrieval ambiguity checks and downstream
legal applicability/answer checks remain unchanged.

Reranker failure, missing/non-probability scores and failed dense retrieval keep
the donor's available lexical ranking. Such fallback is visible in the retrieval
route events. It does not replace v8's generator or verifier with another model.

## Configuration

| Variable | Default / behavior |
|---|---|
| `TOP_K` | 12, clamped to 4–30 as in the donor |
| `RETRIEVAL_CANDIDATES` | 64, clamped to 16–256; at least TOP_K |
| `CONDITIONAL_DENSE_THRESHOLD` | 0.5 |
| `CONTEXT_MAX_CHARS` | 18000, donor record-packing budget |
| `CONTEXT_MAX_SOURCES` | 18, including expansions |
| `RETRIEVAL_MODE` | `conditional`; legacy `hybrid` is accepted as this same donor strategy; `bm25` is an explicit offline/lexical-only mode |
| `RERANKER_TOP_N` | Retained in v8 configuration but NOT used by this donor strategy; the actual rerank request asks for the entire candidate pool |
| `EMB_MODEL` | `BAAI/bge-m3` |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-8B` |
| `CHUNKER` | Ignored as before; platform `common` cannot block startup |

Other v8 settings are unchanged, including MAX_TOKENS=8192, TEMPERATURE=0,
MAX_WORKERS=8, REQUEST_TIMEOUT=90 and RUN_BUDGET_SECONDS=1100.
Platform environment values override defaults. In particular, explicitly set
`TOP_K=12` when comparing against the donor defaults: an inherited platform
TOP_K=40 is clamped to 30, just as the donor would do, and is not equivalent.
The v8 startup log still displays its stored `rerank_top_n` setting; the actual
donor pool size and route are recorded in retrieval events and HTTP payloads.

## Data and defenses

The bundle contains genuine cached BGE-M3 document embeddings from the existing
benchmark seed, model revision `5617a9f61b028005a4858fdac845db406aefb181`.
All 342 search-text hashes match the donor exactly. Only document vectors are
included; question vectors, answers, scores, knowledge-base files and secrets
are excluded. Cache identity includes model name and every search-text hash.
A changed corpus/model cannot reuse the bundled index and requires rebuilding
through the configured embedding API.

v8 source audit still runs on its original audit chunks, allowing the previously
validated audit cache to remain valid. Instruction spans are transferred by
document identity and original offsets to selected donor chunks. The raw KB is
not edited or stripped. The donor's sanitized representation is used only for
retrieval/packing. SQLite retains both audit chunks and donor citation chunks,
so original relation references and selected source IDs are resolvable.

## Verification

`verify_retriever35.py REPORT.json` materializes the actual donor commit through
Git and compares it with the adapter. It checks exact search text/record equality,
AST equality of transferred ranking methods, selected IDs/order and service-call
order for 58 original cases plus 12 alternate-route cases. The reranker is
scripted: these 70 checks establish implementation parity, not answer quality.

`test_retriever35.py` checks offsets, vectors, cache invalidation, instruction-span
transfer and rejection, concurrent isolation, original reranker payloads and
separate endpoint credentials. v8 tests cover the unchanged answer pipeline,
failure handling, sessions and batch deadline. `test_package.py` executes the
extracted ZIP using a local HTTP substitute in three platform configurations.
No paid model calls are part of these checks.

A live LLM/platform evaluation is still required to measure score and runtime
for this combination. Neither 35 points nor 690 seconds is guaranteed.
