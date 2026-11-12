"""Platform entry point. Keep imports in root-level Python files."""
import os
from legal_core import run_submission


if __name__ == "__main__":
    os.environ["USE_EMBEDDINGS"] = "1"
    os.environ["RETRIEVAL_MODE"] = "conditional"
    os.environ.setdefault("ANSWER_REVIEW_ENABLED", "1")
    # Isolated experiment: keep review, restore the separate dense fallback.
    os.environ["CONDITIONAL_DENSE_MERGE"] = "0"
    run_submission()
