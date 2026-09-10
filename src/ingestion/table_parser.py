from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import warnings
import io
import json

warnings.filterwarnings("ignore")

def clean_sec_table(df: pd.DataFrame) -> str:
    """
    Advanced data cleaning pipeline for SEC financial tables.
    """
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df = df.fillna("")
    
    raw_cleaned_rows = []
    max_cols = 0
    
    for _, row in df.iterrows():
        row_values = row.astype(str).tolist()
        deduped_row = []
        pending_prefix = ""
        
        for i, val in enumerate(row_values):
            val = val.strip().replace('\n', ' ').replace('\r', '')
            
            if i == 0 and not val and not raw_cleaned_rows:
                val = "Metric/Region"
                
            if val:
                if val == "$":
                    pending_prefix = "$"
                    continue
                if val == "%":
                    if deduped_row:
                        deduped_row[-1] = f"{deduped_row[-1]}%"
                    continue

                if not deduped_row or val != deduped_row[-1]:
                    if deduped_row and val == f"{deduped_row[-1]}.0":
                        continue
                    if deduped_row and deduped_row[-1] == f"{val}.0":
                        deduped_row[-1] = f"{pending_prefix}{val}"
                        pending_prefix = ""
                        continue
                        
                    deduped_row.append(f"{pending_prefix}{val}")
                    pending_prefix = ""
                    
        if deduped_row:
            raw_cleaned_rows.append(deduped_row)
            if len(deduped_row) > max_cols:
                max_cols = len(deduped_row)

    cleaned_rows = []
    for idx, row in enumerate(raw_cleaned_rows):
        if idx == 0 and len(row) == max_cols - 1:
            row.insert(0, "Metric/Region")
        cleaned_rows.append(" | ".join(row))
        
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
    print(f"[*] Processing SEC submission: {filepath.name}")
    
    # Extract metadata directly from the folder structure
    # Path format: data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt
    accession_number = filepath.parent.name
    filing_type = filepath.parents[1].name
    ticker = filepath.parents[2].name
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml")
    
    html_tables = soup.find_all("table")
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
                        
                        # Create the metadata-enriched JSON payload
                        payload = {
                            "document_id": f"{ticker}_{filing_type}_{accession_number}",
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "table_index": valid_tables_found,
                            "content": clean_markdown
                        }
                        
                        # Save to data/processed directory
                        output_filename = f"{ticker}_{filing_type}_table_{valid_tables_found}.json"
                        output_path = output_dir / output_filename
                        
                        with open(output_path, "w", encoding="utf-8") as json_file:
                            json.dump(payload, json_file, indent=4)
                            
                        print(f"[✓] Saved {output_filename} to {output_dir}")
                        
                        # Stop after finding the first 3 CORE financial statements
                        if valid_tables_found >= 3:
                            break
                            
        except ValueError:
            continue
            
    print(f"\n[+] Extraction complete. {valid_tables_found} tables saved.")

if __name__ == "__main__":
    test_file = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt")
    processed_dir = Path("data/processed")
    
    if test_file.exists():
        extract_and_save_tables(test_file, processed_dir)
    else:
        print("File not found. Please verify the file path.")