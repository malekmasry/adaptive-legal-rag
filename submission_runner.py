"""Deadline-aware CSV runner with atomic checkpoints and per-ID telemetry."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter

from rag_telemetry import CURRENT, DEADLINE, Trace
from single_pass import irac_notice


def read_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("questions.csv must contain id,question")
        questions = []
        for row in reader:
            if None in row or row.get("id") is None or row.get("question") is None:
                raise ValueError("Malformed CSV row")
            questions.append((row["id"].strip(), row["question"].strip()))
    if not questions or any(not item_id for item_id, _ in questions):
        raise ValueError("Empty question set or ID")
    if len({item_id for item_id, _ in questions}) != len(questions):
        raise ValueError("Duplicate question ID")
    return questions


def checkpoint(path: Path, questions: list, answers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=".submission-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            writer.writerows((item_id, answer) for (item_id, _), answer in zip(questions, answers))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_submission() -> None:
    # Imported here so legal_core.run_submission remains a compatible entry point.
    from legal_core import Answer, LegalAssistant, find_archive, integer_setting, setting

    started = perf_counter()
    timeout = integer_setting("BATCH_TIMEOUT_SECONDS", 1080, 1, 1200)
    reserve = min(integer_setting("BATCH_WRITE_RESERVE_SECONDS", 5, 0, 60), timeout / 4)
    deadline = started + timeout - reserve
    data_dir = Path(setting("DATA_DIR", default="data"))
    questions = read_questions(data_dir / "questions.csv")
    output = Path(setting("OUTPUT_CSV", default="submission.csv"))
    metrics_path = Path(setting("METRICS_PATH", default=str(output.with_suffix(".metrics.jsonl"))))
    if metrics_path.resolve() == output.resolve() or output.resolve() == (data_dir / "questions.csv").resolve():
        raise ValueError("Output, metrics and input paths must be distinct")
    unfinished = irac_notice("Обработка вопроса не завершена.",
                             "Ответ не получен до завершения запуска. Юридический вывод не сформирован.")
    answers = [unfinished] * len(questions)
    records = {}
    checkpoint(output, questions, answers)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    batch_token = DEADLINE.set(deadline)
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics:
            def record(index, answer, *, persist=True):
                answers[index] = answer.text.replace("\x00", "").strip() or unfinished
                diagnostic_keys = ("pipeline", "total_ms", "search_total_ms", "timings_ms", "pipeline_calls",
                                   "prompt_tokens", "completion_tokens", "usage_complete", "api_calls", "events")
                info = {key: answer.diagnostics[key] for key in diagnostic_keys if key in answer.diagnostics}
                # Legacy adaptive events can contain draft excerpts. Keep only
                # compact operational fields in the default batch log.
                info["events"] = [{key: event[key] for key in (
                    "stage", "status", "type", "count", "source_ids", "irac_ok", "unknown_citations",
                    "has_citations", "nonliteral_quotes", "quarantined_lines", "dense_used", "characters", "skipped_ids",
                    "route", "threshold", "rerank_score", "dense_rerank_score",
                    "query_count", "unique_candidates", "selected_candidates", "error_type") if key in event}
                                  for event in info.get("events", [])]
                info.update({"id": questions[index][0], "status": answer.status,
                             "source_ids": [source.id for source in answer.sources]})
                records[index] = info
                if persist:
                    checkpoint(output, questions, answers)
                metrics.write(json.dumps(info, ensure_ascii=False) + "\n")
                metrics.flush()

            init_trace = Trace(deadline_seconds=min(timeout, integer_setting("INDEX_TIMEOUT_SECONDS", 120, 1, 600)),
                               stage="indexing")
            init_token = CURRENT.set(init_trace)
            try:
                engine = LegalAssistant(find_archive(data_dir))
            except Exception as exc:
                for index in range(len(questions)):
                    record(index, Answer(unfinished, [], "technical_error",
                                         {"events": [{"stage": "initialization", "type": type(exc).__name__}]}), persist=False)
                checkpoint(output, questions, answers)
                print(json.dumps({"stage": "initialization", "error_type": type(exc).__name__}), file=sys.stderr)
                return
            finally:
                CURRENT.reset(init_token)
            metrics.write(json.dumps({"stage": "initialization", **init_trace.public(),
                                      "documents": len(getattr(engine, "documents", [])),
                                      "sources": len(getattr(engine, "sources", [])),
                                      "links": len(getattr(engine, "links", [])),
                                      "retrieval_mode": engine.retrieval_mode,
                                      "conditional_threshold": engine.conditional_threshold,
                                      "dense_index_ready": engine.vectors is not None,
                                      "dense_multi_query": setting("DENSE_MULTI_QUERY", default="0") == "1",
                                      "ingestion_warning_counts": dict(Counter(
                                          item.get("reason", "unknown") for item in getattr(engine, "ingestion_warnings", [])))},
                                     ensure_ascii=False) + "\n")
            metrics.flush()

            def work(index):
                token = DEADLINE.set(deadline)
                try:
                    if perf_counter() >= deadline:
                        return Answer(unfinished, [], "technical_error", {"events": [{"stage": "batch_deadline"}]})
                    return engine.answer(questions[index][1])
                except Exception as exc:
                    return Answer(unfinished, [], "technical_error",
                                  {"events": [{"stage": "worker", "type": type(exc).__name__}]})
                finally:
                    DEADLINE.reset(token)

            workers = integer_setting("MAX_WORKERS", 3, 1, 8)
            pool = ThreadPoolExecutor(max_workers=workers)
            active, next_index = {}, 0
            try:
                while (next_index < len(questions) or active) and perf_counter() < deadline:
                    while next_index < len(questions) and len(active) < workers and perf_counter() < deadline:
                        active[pool.submit(work, next_index)] = next_index
                        next_index += 1
                    if not active:
                        break
                    done, _ = wait(active, timeout=max(0, deadline - perf_counter()), return_when=FIRST_COMPLETED)
                    for future in done:
                        record(active.pop(future), future.result())
                # Capture completed results even if the scheduler woke at the deadline.
                for future, index in list(active.items()):
                    if future.done() and not future.cancelled():
                        record(index, future.result())
                        del active[future]
            finally:
                for future in active:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
            for index in range(len(questions)):
                if index not in records:
                    record(index, Answer(unfinished, [], "technical_error", {"events": [{"stage": "batch_deadline"}]}), persist=False)
            checkpoint(output, questions, answers)
            counts = {status: sum(item["status"] == status for item in records.values())
                      for status in sorted({item["status"] for item in records.values()})}
            print(json.dumps({"stage": "batch_complete", "rows": len(questions), "statuses": counts,
                              "elapsed_seconds": round(perf_counter() - started, 2)}, ensure_ascii=False), file=sys.stderr)
    finally:
        DEADLINE.reset(batch_token)
