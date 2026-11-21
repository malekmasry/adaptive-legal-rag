import os
from legal_core import run_submission
if __name__ == "__main__":
    os.environ["USE_EMBEDDINGS"]="0"
    os.environ["RERANKER_MODEL"]=""
    os.environ["RETRIEVAL_MODE"]="hybrid"
    os.environ["TOP_K"]="6"
    os.environ["CONTEXT_MAX_SOURCES"]="6"
    os.environ["CONTEXT_MAX_CHARS"]="7000"
    run_submission()
