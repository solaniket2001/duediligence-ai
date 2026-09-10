import os
from pathlib import Path
from sec_edgar_downloader import Downloader

# Set the base directory to store our raw 10-K filings.
# I'm placing this in the data/raw folder we created earlier so it stays organized.
DATA_DIR = Path("data/raw")

def download_filing(ticker: str, filing_type: str = "10-K", limit: int = 1) -> Path:
    """
    Downloads SEC filings for a specific stock ticker.
    
    Note: We need to pass a specific user agent to comply with SEC Edgar's 
    programmatic fair access policy, otherwise our IP address will get rate-limited 
    or permanently blocked.
    """
    # Make sure the raw directory exists before trying to save files into it
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # SEC requires a "CompanyName" and an "Email" to track API usage.
    # Using my own details here to adhere to their compliance rules.
    company_name = "DueDiligenceAI_Project"
    email_address = "aniket2312001@gmail.com"
    
    print(f"[*] Initializing downloader for {ticker}...")
    
    # Initialize the SEC downloader with our credentials and save path
    dl = Downloader(company_name, email_address, DATA_DIR)
    
    # Fetch the actual filings. 
    # limit=1 means we only grab the single most recent report to save disk space.
    print(f"[*] Fetching the latest {limit} {filing_type} filing(s)...")
    dl.get(filing_type, ticker, limit=limit)
    
    print(f"[+] Successfully downloaded {filing_type} for {ticker} into {DATA_DIR.resolve()}")
    return DATA_DIR

if __name__ == "__main__":
    # Testing it out with Apple's latest 10-K filing to ensure the connection works
    download_filing("AAPL", filing_type="10-K", limit=1)