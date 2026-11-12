"""Run one IRAC question without a web server; credentials stay in the process."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from legal_core import LegalAssistant, find_archive, setting


def main() -> int:
    parser = argparse.ArgumentParser(description="Legal RAG: single-pass IRAC with source references")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--question-file", type=Path)
    group.add_argument("--chat", action="store_true", help="In-memory conversation; /new clears it, /exit ends it")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--pipeline", choices=("single_pass", "adaptive"), default="single_pass")
    parser.add_argument("--embeddings", action="store_true", default=None, help="Enable configured embedding API")
    parser.add_argument("--ask-key", action="store_true", help="Read an API key without displaying or saving it")
    parser.add_argument("--trace", type=Path, help="Save private diagnostic JSON, including source excerpts")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    if args.trace and args.trace.exists():
        raise SystemExit("Refusing to overwrite an existing trace")
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    os.environ["PIPELINE_MODE"] = args.pipeline
    if args.ask_key:
        os.environ["LLM_API_KEY"] = getpass.getpass("API key (not saved): ")
    if args.chat and args.trace:
        parser.error("--trace is only supported for one-question runs")
    archive = args.archive or find_archive(Path(setting("DATA_DIR", default="data")))
    engine = LegalAssistant(archive, use_embeddings=args.embeddings)
    if args.chat:
        session = engine.new_session()
        print("Диалог начат. /new — очистить историю; /exit — выйти.")
        while True:
            try:
                question = input("Вы: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question == "/exit":
                break
            if question == "/new":
                session.clear()
                print("История очищена.")
                continue
            if question:
                result = session.ask(question)
                print(result.text)
                if args.metrics:
                    print(json.dumps(result.diagnostics, ensure_ascii=False), file=sys.stderr)
        return 0
    question = args.question if args.question is not None else args.question_file.read_text(encoding="utf-8-sig")
    result = engine.answer(question)
    print(result.text)
    if args.metrics:
        diagnostics = result.diagnostics
        print(json.dumps({key: diagnostics.get(key) for key in
                          ("total_ms", "search_total_ms", "timings_ms", "pipeline_calls", "prompt_tokens",
                           "completion_tokens", "usage_complete")}, ensure_ascii=False), file=sys.stderr)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps(result.public(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result.status == "technical_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
