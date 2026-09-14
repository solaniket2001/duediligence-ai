import os
import re
import io
import json
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("table_parser")

def clean_sec_table(df: pd.DataFrame) -> str:
    cleaned_rows = []
    for _, row in df.iterrows():
        row_values = [str(val).strip() for val in row.values if pd.notna(val)]
        processed_row = []
        for i, val in enumerate(row_values):
            val = val.replace('\n', ' ').replace('\r', '')
            if i == 0 and not val and not cleaned_rows:
                val = "Metric/Region"
            if val in ["—", "–", "-"]:
                val = "0"
            # Normalize negative parentheses (6) -> -6
            val = re.sub(r'^\(([\d,\.]+)(%?)\)(%?)$', r'-\1\2\3', val)
            if val:
                processed_row.append(val)
        if len(processed_row) >= 2:
            cleaned_rows.append(" | ".join(processed_row))
    return "\n".join(cleaned_rows)

def extract_and_save_sec_data(filepath: Path, output_dir: Path):
    if not filepath.exists():
        logger.error(f"Filing not found at {filepath}")
        return

    logger.info(f"Processing SEC submission: {filepath.name}")
    accession_number = filepath.parent.name
    filing_type = filepath.parents[1].name
    ticker = filepath.parents[2].name

    try:
        year_suffix = accession_number.split('-')[1]
        year = f"20{year_suffix}" if len(year_suffix) == 2 else "2025"
    except IndexError:
        year = "2025"

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. EXTRACT FINANCIAL TABLES
    
    tables = soup.find_all('table')
    valid_tables_found = 0
    keywords = ["iPhone", "Mac", "Services", "Americas", "Gross Margin", "Net sales", "Total net sales"]

    for table in tables:
        table_text = table.get_text()
        matches = sum(1 for kw in keywords if kw.lower() in table_text.lower())
        if matches >= 2:
            try:
                # Wrap HTML string in io.StringIO to satisfy modern Pandas
                dfs = pd.read_html(io.StringIO(str(table)))
                if dfs:
                    df = dfs[0]
                    clean_md = clean_sec_table(df)
                    if len(clean_md.splitlines()) >= 3:
                        valid_tables_found += 1
                        payload = {
                            "document_id": f"{ticker}_{filing_type}_{year}_table_{valid_tables_found}",
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "year": year,
                            "chunk_type": "table",
                            "content": clean_md
                        }
                        with open(output_dir / f"{payload['document_id']}.json", "w", encoding="utf-8") as out:
                            json.dump(payload, out, indent=4)
                        logger.info(f"Successfully saved table: {payload['document_id']}")
            except Exception as e:
                logger.debug(f"Table parse skipped: {e}")
                continue

    # 2. EXTRACT TARGETED RISK & BUSINESS DISCLOSURES
    
    paragraphs = soup.find_all('p')
    valid_texts_found = 0
    seen_texts = set()
    risk_markers = ["risk", "adversely", "competition", "regulatory", "supply chain", "economic conditions", "litigation"]

    for p in paragraphs:
        text = p.get_text(separator=' ', strip=True)
        if 200 <= len(text) <= 2000 and text not in seen_texts:
            if any(marker in text.lower() for marker in risk_markers):
                seen_texts.add(text)
                valid_texts_found += 1
                payload = {
                    "document_id": f"{ticker}_{filing_type}_{year}_text_{valid_texts_found}",
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "year": year,
                    "chunk_type": "text",
                    "content": text
                }
                with open(output_dir / f"{payload['document_id']}.json", "w", encoding="utf-8") as out:
                    json.dump(payload, out, indent=4)
                if valid_texts_found >= 30:
                    break

    logger.info(f"Extraction Complete: Saved {valid_tables_found} tables and {valid_texts_found} curated risk texts.")

if __name__ == "__main__":
    SEC_FILE = Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt")
    PROCESSED_DIR = Path("data/processed")
    extract_and_save_sec_data(SEC_FILE, PROCESSED_DIR)