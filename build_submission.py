"""Build a self-contained submission ZIP; credentials and corpus are excluded."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = ("solution.py", "legal_core.py", "adaptive_pipeline.py", "hallugraph.py",
         "rag_safety.py", "rag_telemetry.py", "single_pass.py", "submission_runner.py",
         "answer_review.py", "requirements.txt")
TARGET = ROOT / "dist" / "submission.zip"


def main() -> None:
    TARGET.parent.mkdir(exist_ok=True)
    with ZipFile(TARGET, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
    with ZipFile(TARGET) as archive:
        assert archive.namelist() == list(FILES)
        assert archive.testzip() is None
    print(f"{TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
