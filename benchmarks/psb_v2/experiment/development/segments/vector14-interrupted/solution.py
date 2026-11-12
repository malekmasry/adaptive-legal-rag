"""Platform entry point. Keep imports in root-level Python files."""
import os
from legal_core import run_submission


if __name__ == "__main__":
    os.environ["USE_EMBEDDINGS"] = "1"
    run_submission()
