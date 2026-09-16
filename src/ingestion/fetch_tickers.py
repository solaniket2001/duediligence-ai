# src/ingestion/fetch_tickers.py
import requests
import json
import os

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "DueDiligenceAI AdminContact@domain.com"}
LOCAL_FILE = "data/company_tickers.json"

def download_tickers():
    os.makedirs("data", exist_ok=True)
    print(f"📥 Downloading SEC company tickers from {SEC_TICKERS_URL}...")
    
    response = requests.get(SEC_TICKERS_URL, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        with open(LOCAL_FILE, "w") as f:
            json.dump(data, f)
        print(f"✅ Successfully saved {len(data)} tickers locally to {LOCAL_FILE}!")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_tickers()