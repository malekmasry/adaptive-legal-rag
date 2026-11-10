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
    parser = argparse.ArgumentParser(description="Adaptive legal RAG: lexical retrieval and verified IRAC")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--question-file", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
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
    if args.ask_key:
        os.environ["LLM_API_KEY"] = getpass.getpass("API key (not saved): ")
    archive = args.archive or find_archive(Path(setting("DATA_DIR", default="data")))
    question = args.question if args.question is not None else args.question_file.read_text(encoding="utf-8-sig")
    engine = LegalAssistant(archive, use_embeddings=False)
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
