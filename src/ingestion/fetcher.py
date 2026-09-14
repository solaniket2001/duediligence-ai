import os
import sys
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader

# Ensure Python can find our parser and indexer modules
sys.path.append(os.getcwd())
from src.ingestion.table_parser import extract_and_save_sec_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sec_fetcher")

def ingest_new_ticker(ticker: str, filing_year: str = "2025"):
    logger.info(f"Starting automated ingestion pipeline for {ticker} (Year: {filing_year})...")
    
    # 1. SEC requires a user agent string (Company Name & Contact Email)
    # Using personal educational placeholder format required by SEC EDGAR API
    dl = Downloader("HochschuleSchmalkalden", "student.aniket@hs-sm.de")
    
    raw_dir = Path("data/raw/sec-edgar-filings")
    
    # 2. Download the 10-K filing from SEC EDGAR
    logger.info(f"Querying SEC EDGAR database for {ticker} 10-K...")
    try:
        dl.get("10-K", ticker.upper(), amount=1, download_details=True)
    except Exception as e:
        logger.error(f"Failed to download filing from SEC: {e}")
        return False
        
    # 3. Locate the freshly downloaded file path dynamically
    ticker_path = raw_dir / ticker.upper() / "10-K"
    if not ticker_path.exists():
        logger.error(f"Download directory for {ticker} not found.")
        return False
        
    # Find the accession folder (e.g., 0000320193-25-000079)
    accession_folders = [f for f in ticker_path.iterdir() if f.is_dir()]
    if not accession_folders:
        logger.error(f"No accession folders found for {ticker}.")
        return False
        
    latest_accession = sorted(accession_folders)[-1]
    sec_file_path = latest_accession / "full-submission.txt"
    
    if not sec_file_path.exists():
        logger.error(f"full-submission.txt not found for {ticker}.")
        return False
        
    logger.info(f"Successfully downloaded filing to: {sec_file_path}")
    
    # 4. Trigger the Table Parser on the new file
    processed_dir = Path("data/processed")
    logger.info("Executing table and text extraction...")
    extract_and_save_sec_data(sec_file_path, processed_dir)
    
    logger.info(f"Ingestion complete for {ticker}. Run `python src/retrieval/indexer.py` to update the vector database.")
    return True

if __name__ == "__main__":
    # Example CLI usage: python src/ingestion/fetcher.py MSFT 2025
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    year_arg = sys.argv[2] if len(sys.argv) > 2 else "2025"
    ingest_new_ticker(ticker_arg, year_arg)