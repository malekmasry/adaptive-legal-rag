"""Platform entry point. Keep imports in root-level Python files."""
import os
from legal_core import run_submission


if __name__ == "__main__":
    os.environ["USE_EMBEDDINGS"] = "1"
    os.environ["RETRIEVAL_MODE"] = "conditional"
    os.environ.setdefault("ANSWER_REVIEW_ENABLED", "1")
    os.environ.setdefault("CONDITIONAL_DENSE_MERGE", "1")
    run_submission()
