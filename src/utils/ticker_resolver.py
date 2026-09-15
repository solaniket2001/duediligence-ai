import difflib
import json
import os

LOCAL_FILE = "data/company_tickers.json"

# We now map directly to the final Tuple so we don't have to rely on the SEC JSON!
COMMON_ALIASES = {
    "FACEBOOK": ("META", "Meta Platforms, Inc."),
    "FB": ("META", "Meta Platforms, Inc."),
    "GOOGLE": ("GOOGL", "Alphabet Inc."),
    "SQUARE": ("0001512673", "Block, Inc."),
    "TWITTER": ("X", "X Corp."),
}

def get_sec_tickers():
    if not os.path.exists(LOCAL_FILE):
        raise FileNotFoundError(f"Missing {LOCAL_FILE}. Please run 'python src/ingestion/fetch_tickers.py' first.")
        
    with open(LOCAL_FILE, "r") as f:
        return json.load(f)

def resolve_ticker(user_input):
    if not user_input or not user_input.strip():
        return None, None

    cleaned_input = user_input.strip().upper()
    
    # 0. Instantly return known aliases (Bypasses the SEC JSON)
    if cleaned_input in COMMON_ALIASES:
        return COMMON_ALIASES[cleaned_input]

    tickers_data = get_sec_tickers()
    
    # 1. Exact Ticker Match
    for entry in tickers_data.values():
        if entry["ticker"] == cleaned_input:
            return entry["ticker"], entry["title"]
            
    # 2. Exact Title Match
    for entry in tickers_data.values():
        if entry["title"].upper() == cleaned_input:
            return entry["ticker"], entry["title"]
            
    # 3. Fuzzy Title Match
    titles = {entry["title"].upper(): entry for entry in tickers_data.values()}
    matches = difflib.get_close_matches(cleaned_input, titles.keys(), n=1, cutoff=0.5)
    
    if matches:
        best_match = titles[matches[0]]
        return best_match["ticker"], best_match["title"]
        
    return None, None