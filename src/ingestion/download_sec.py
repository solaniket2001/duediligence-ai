import os
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("sec_ingestion")

DATA_DIR = Path("data/raw")

# 2. Implement Exponential Backoff Retries
# If the SEC server drops the connection, wait 2 seconds, then 4, then stop after 3 total attempts.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def download_filing(ticker: str, filing_type: str = "10-K", limit: int = 1) -> Path:
    """
    Downloads SEC filings with network resilience and structured logging.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    company_name = "DueDiligenceAI_Project"
    email_address = "aniket2312001@gmail.com"
    
    logger.info(f"Initializing SEC downloader for ticker: {ticker}")
    dl = Downloader(company_name, email_address, DATA_DIR)
    
    logger.info(f"Attempting to fetch the latest {limit} {filing_type} filing(s)...")
    dl.get(filing_type, ticker, limit=limit)
    
    logger.info(f"Successfully downloaded {filing_type} for {ticker} into {DATA_DIR.resolve()}")
    return DATA_DIR

if __name__ == "__main__":
    try:
        download_filing("AAPL", filing_type="10-K", limit=1)
    except Exception as e:
        logger.error(f"Failed to download SEC data after 3 attempts: {e}")