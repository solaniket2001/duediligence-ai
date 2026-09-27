import glob
import subprocess
import sys
import os
from pathlib import Path

from src.ingestion.table_parser import extract_and_save_sec_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.ingest <TICKER>")
        sys.exit(1)

    # Handle the case where a flag was passed (e.g., --ticker AMD)
    if sys.argv[1] == "--ticker" and len(sys.argv) > 2:
        ticker = sys.argv[2].upper()
    else:
        ticker = sys.argv[1].upper()

    print(f"🚀 Starting automated ingestion pipeline for {ticker}...\n")

    # Force Python to recognize the root directory
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # 1. Download SEC 10-K (Run as a module with -m)
    print(f"--- [1/3] Downloading 10-K for {ticker} ---")
    subprocess.run(
        [sys.executable, "-m", "src.ingestion.download_sec", "--ticker", ticker],
        env=env,
        check=True,
    )

    # 2. Parse HTML tables and text into JSON
    print(f"\n--- [2/3] Parsing HTML to JSON ---")
    
    # Check for both new and old naming conventions from the SEC EDGAR API
    search_pattern_txt = f"data/raw/sec-edgar-filings/{ticker}/10-K/*/full-submission.txt"
    search_pattern_html = f"data/raw/sec-edgar-filings/{ticker}/10-K/*/primary-document.html"
    
    files = glob.glob(search_pattern_txt) + glob.glob(search_pattern_html)

    if not files:
        print(f"❌ Error: Could not find downloaded HTML or TXT for {ticker}!")
        sys.exit(1)

    # Sort to pick the latest filing directory if multiple exist
    target_file = sorted(files)[-1]
    print(f"Found filing: {target_file}")
    
    # Send it to the parser
    extract_and_save_sec_data(Path(target_file), Path("data/processed"))

    # 3. Vector Indexing into ChromaDB (Run as a module with -m)
    print(f"\n--- [3/3] Indexing into ChromaDB ---")
    subprocess.run(
        [sys.executable, "-m", "src.retrieval.indexer"],
        env=env,
        check=True,
    )

    print(f"\n✅ {ticker} is now fully ingested and ready for the Due Diligence API!")

if __name__ == "__main__":
    main()