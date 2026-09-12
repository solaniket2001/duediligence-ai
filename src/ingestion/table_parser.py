from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import warnings
import io
import json
import logging
import re

warnings.filterwarnings("ignore")

# 1. Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("table_parser")

def clean_sec_table(df: pd.DataFrame) -> str:
    """Advanced data cleaning pipeline for SEC financial tables."""
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df = df.fillna("")
    
    raw_cleaned_rows = []
    
    for _, row in df.iterrows():
        row_values = row.astype(str).tolist()
        deduped_row = []
        pending_prefix = ""
        
        for i, val in enumerate(row_values):
            val = val.strip().replace('\n', ' ').replace('\r', '')
            
            # 1. Header padding fix
            if i == 0 and not val and not raw_cleaned_rows:
                val = "Metric/Region"
                
            # 2. Normalize SEC financial dashes to "0"
            if val in ["—", "–", "-"]:
                val = "0"
                
            # 3. NEW: Convert accounting parentheses to standard minus signs
            # Safely extracts the number, turning "(6)" or "(6%)" into "-6" or "-6%"
            val = re.sub(r'^\(([\d,\.]+)(%?)\)(%?)$', r'-\1\2\3', val)
                
            if val:
                if val == "$":
                    pending_prefix = "$"
                    continue
                if val == "%":
                    if deduped_row:
                        deduped_row[-1] = f"{deduped_row[-1]}%"
                    continue

                if not deduped_row or val != deduped_row[-1]:
                    if deduped_row:
                        # 2. Float Fix: Strip prefix symbols for accurate float comparison
                        prev_clean = deduped_row[-1].replace("$", "")
                        if val == f"{prev_clean}.0":
                            continue
                        if prev_clean == f"{val}.0":
                            deduped_row[-1] = f"{pending_prefix}{val}"
                            pending_prefix = ""
                            continue
                        
                    deduped_row.append(f"{pending_prefix}{val}")
                    pending_prefix = ""
                    
        if deduped_row:
            raw_cleaned_rows.append(deduped_row)

    cleaned_rows = [" | ".join(row) for row in raw_cleaned_rows]
    return "\n".join(cleaned_rows)

def is_core_financial_table(df: pd.DataFrame) -> bool:
    """Evaluates semantic content to find core financial statements."""
    text_dump = df.to_string().lower()
    financial_keywords = [
        "total net sales", "gross margin", "operating expenses", 
        "net income", "total assets", "total liabilities", 
        "cash equivalents", "retained earnings"
    ]
    has_keywords = any(keyword in text_dump for keyword in financial_keywords)
    digit_count = sum(c.isdigit() for c in text_dump)
    
    if has_keywords and digit_count > 50:
        return True
    return False

def extract_and_save_tables(filepath: Path, output_dir: Path):
    logger.info(f"Processing SEC submission: {filepath.name}")
    
    accession_number = filepath.parent.name
    filing_type = filepath.parents[1].name
    ticker = filepath.parents[2].name
    
    # NEW: Extract the year from the accession number format (CIK-YY-XXXXXX)
    try:
        year_suffix = accession_number.split('-')[1]
        year = f"20{year_suffix}" if len(year_suffix) == 2 else "Unknown"
    except IndexError:
        year = "Unknown"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml")
    
    html_tables = soup.find_all("table")
    logger.info(f"Found {len(html_tables)} total tables. Executing semantic filtering...")
    valid_tables_found = 0
    
    for index, table in enumerate(html_tables):
        try:
            html_string = io.StringIO(str(table))
            df = pd.read_html(html_string)[0]
            
            if df.shape[0] > 5 and df.shape[1] > 2:
                if is_core_financial_table(df):
                    clean_markdown = clean_sec_table(df)
                    
                    if len(clean_markdown.strip()) > 0:
                        valid_tables_found += 1
                        
                        payload = {
                            "document_id": f"{ticker}_{filing_type}_{year}_{accession_number}",
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "year": year, # NEW METADATA FIELD
                            "table_index": valid_tables_found,
                            "content": clean_markdown
                        }
                        
                        output_filename = f"{ticker}_{filing_type}_table_{valid_tables_found}.json"
                        output_path = output_dir / output_filename
                        
                        with open(output_path, "w", encoding="utf-8") as json_file:
                            json.dump(payload, json_file, indent=4)
                            
                        logger.info(f"Saved {output_filename} to {output_dir}")
                        
                        if valid_tables_found >= 3:
                            break
                            
        except ValueError as e:
            # We use logger.debug here so it doesn't clutter the terminal, 
            # but is still available if we need to investigate parsing failures.
            logger.debug(f"Skipping unparseable table {index}: {e}")
            continue
            
    logger.info(f"Extraction complete. {valid_tables_found} core tables processed and saved.")

if __name__ == "__main__":
    test_file = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt")
    processed_dir = Path("data/processed")
    
    if test_file.exists():
        extract_and_save_tables(test_file, processed_dir)
    else:
        logger.error(f"File not found: {test_file}. Please verify the file path.")