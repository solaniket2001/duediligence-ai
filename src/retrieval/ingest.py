import glob
import subprocess
import sys
import os
from pathlib import Path

from src.ingestion.table_parser import extract_and_save_sec_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.ingest <TICKER>", flush=True)
        sys.exit(1)

    if sys.argv[1] == "--ticker" and len(sys.argv) > 2:
        ticker = sys.argv[2].upper().strip()
    else:
        ticker = sys.argv[1].upper().strip()

    print(f"Starting automated ingestion pipeline for {ticker}...", flush=True)

    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONUNBUFFERED"] = "1"

    # 1. Download SEC 10-K
    print(f"[1/3] Downloading SEC 10-K filing for {ticker}...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "-m", "src.ingestion.download_sec", "--ticker", ticker],
        cwd=str(project_root),
        env=env,
        check=True,
    )

    # 2. Parse HTML tables and text into JSON
    print(f"[2/3] Parsing SEC HTML tables to JSON for {ticker}...", flush=True)
    search_pattern_txt = str(project_root / f"data/raw/sec-edgar-filings/{ticker}/10-K/*/full-submission.txt")
    search_pattern_html = str(project_root / f"data/raw/sec-edgar-filings/{ticker}/10-K/*/primary-document.html")

    files = glob.glob(search_pattern_txt) + glob.glob(search_pattern_html)
    if not files:
        print(f"[Error] Could not find downloaded HTML or TXT for {ticker}!", flush=True)
        sys.exit(1)

    target_file = sorted(files)[-1]
    print(f"Found filing: {target_file}", flush=True)

    extract_and_save_sec_data(Path(target_file), project_root / "data/processed")

    # 3. Vector Indexing into ChromaDB
    print(f"[3/3] Indexing {ticker} chunks into ChromaDB...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "-m", "src.retrieval.indexer", "--ticker", ticker],
        cwd=str(project_root),
        env=env,
        check=True,
    )

    print(f"[Success] {ticker} is now fully ingested and indexed!", flush=True)


if __name__ == "__main__":
    main()