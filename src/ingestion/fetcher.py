import os
import sys
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader

sys.path.append(os.getcwd())
from src.ingestion.table_parser import extract_and_save_sec_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sec_fetcher")

def ingest_new_ticker(ticker: str, filing_year: str = "2025", filing_type: str = "annual"):
    logger.info(f"Starting automated ingestion pipeline for {ticker} (Year: {filing_year}, Type: {filing_type})...")

    # SEC EDGAR Downloader initialization
    dl = Downloader("HochschuleSchmalkalden", "student.aniket@hs-sm.de", "data/raw")
    raw_dir = Path("data/raw/sec-edgar-filings") / ticker.upper()

    downloaded_form = None

    # 1. Determine form to download
    if filing_type.lower() == "quarterly":
        logger.info(f"Querying SEC EDGAR database for {ticker} 10-Q...")
        try:
            dl.get("10-Q", ticker.upper(), limit=1, download_details=True)
            downloaded_form = "10-Q"
        except Exception as e:
            logger.error(f"Failed to download 10-Q: {e}")
            return False
    else:
        # Annual: Try 10-K first
        logger.info(f"Querying SEC EDGAR database for {ticker} 10-K...")
        try:
            count = dl.get("10-K", ticker.upper(), limit=1, download_details=True)
            if count and count > 0:
                downloaded_form = "10-K"
        except Exception as e:
            logger.warning(f"Error checking 10-K: {e}")

        # Fallback to 20-F if 10-K was not found (Foreign Private Issuers)
        form_10k_path = raw_dir / "10-K"
        if not form_10k_path.exists() or not list(form_10k_path.iterdir()):
            logger.warning(f"No 10-K found for {ticker}. Attempting foreign issuer fallback (20-F)...")
            try:
                dl.get("20-F", ticker.upper(), limit=1, download_details=True)
                downloaded_form = "20-F"
            except Exception as e:
                logger.error(f"Failed to download 20-F fallback: {e}")
                return False
        else:
            downloaded_form = "10-K"

    # 2. Locate the downloaded file dynamically
    form_path = raw_dir / downloaded_form
    if not form_path.exists():
        logger.error(f"Download directory {form_path} not found.")
        return False

    accession_folders = [f for f in form_path.iterdir() if f.is_dir()]
    if not accession_folders:
        logger.error(f"No accession folders found for {ticker} under {downloaded_form}.")
        return False

    latest_accession = sorted(accession_folders)[-1]
    sec_file_path = latest_accession / "full-submission.txt"

    if not sec_file_path.exists():
        logger.error(f"full-submission.txt not found for {ticker}.")
        return False

    logger.info(f"Successfully downloaded filing ({downloaded_form}) to: {sec_file_path}")

    # 3. Trigger Table & Text Extraction
    processed_dir = Path("data/processed")
    logger.info("Executing table and text extraction...")
    extract_and_save_sec_data(sec_file_path, processed_dir)

    logger.info(f"Ingestion complete for {ticker}. Run `python src/retrieval/indexer.py` to update the vector database.")
    return True

if __name__ == "__main__":
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    year_arg = sys.argv[2] if len(sys.argv) > 2 else "2025"
    type_arg = sys.argv[3] if len(sys.argv) > 3 else "annual"
    ingest_new_ticker(ticker_arg, year_arg, type_arg)