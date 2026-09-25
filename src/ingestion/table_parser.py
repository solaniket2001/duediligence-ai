import io
import json
import logging
import re
from html import escape as html_escape
from pathlib import Path

import pandas as pd
from lxml import html
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("table_parser")


def clean_sec_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
        
    # 1. Top-left cell padding
    if pd.isna(df.iloc[0, 0]) or str(df.iloc[0, 0]).strip() == "":
        df.iloc[0, 0] = "Metric/Region"
        
    cleaned_rows = []
    
    for i, row in df.iterrows():
        cleaned_row = []
        skip_next = False
        
        for j in range(len(row)):
            if skip_next:
                skip_next = False
                continue
                
            val = str(row.iloc[j]).strip()
            
            # 2. Symbol merging (Merge stray $ or % with the next/previous number)
            if val == "$" and j + 1 < len(row):
                next_val = str(row.iloc[j+1]).strip()
                cleaned_row.append(f"${next_val}")
                skip_next = True
            elif val == "%" and len(cleaned_row) > 0:
                cleaned_row[-1] = f"{cleaned_row[-1]}%"
            else:
                cleaned_row.append(val)
                
        cleaned_rows.append(cleaned_row)

    # 3. Deduplicate headers and data rows
    final_rows = []
    for i, row in enumerate(cleaned_rows):
        seen = set()
        deduped_row = []
        
        for col in row:
            if not col: 
                continue
                
            # Treat "100" and "100.0" as the same value to eliminate phantom floats
            compare_val = col.replace(".0", "").replace("$", "")
            
            if compare_val not in seen:
                deduped_row.append(col)
                seen.add(compare_val)
                
        final_rows.append(" | ".join(deduped_row))
            
    return "\n".join(final_rows)

    # 3. Deduplicate headers (e.g., removing repeated "2025" or "Change")
    final_rows = []
    for i, row in enumerate(cleaned_rows):
        if i == 0:  # Header row deduplication
            seen = set()
            deduped_header = []
            for col in row:
                if col and col not in seen:
                    deduped_header.append(col)
                    seen.add(col)
            final_rows.append(" | ".join(deduped_header))
        else:
            # Filter out empty strings from data rows to match header alignment
            data_row = [col for col in row if col]
            final_rows.append(" | ".join(data_row))
            
    return "\n".join(final_rows)


def _get_preceding_context(element) -> str:
    """Finds the nearest preceding text (like a heading) to provide context for a table."""
    try:
        # Find preceding paragraphs or headings
        preceding = element.xpath("./preceding::*[local-name()='p' or local-name()='div' or local-name()='span' or starts-with(local-name(), 'h')][normalize-space(text())!='']")
        context_parts = []
        # Look backwards through the last 5 elements
        for p in reversed(preceding[-5:]):
            text = re.sub(r"\s+", " ", " ".join(p.itertext())).strip()
            # Only keep substantial text fragments, not page numbers
            if text and len(text) > 8 and not text.isdigit():
                context_parts.append(text)
                if len(context_parts) == 2:  # Grab up to 2 preceding context strings
                    break
        
        if context_parts:
            return " > ".join(reversed(context_parts))
    except Exception:
        pass
    return "Financial Table"


def _extract_10k_document(content: str) -> str:
    documents = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, flags=re.IGNORECASE | re.DOTALL)
    for document in documents:
        if re.search(r"<TYPE>\s*(10-K|10-Q|20-F)(?:\s|<|$)", document, re.IGNORECASE):
            match = re.search(r"<TEXT>(.*?)(?:</TEXT>|$)", document, flags=re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else document.strip()
    return content


def _local_name(element) -> str:
    tag = element.tag
    if not isinstance(tag, str): return ""
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1].lower()


def _is_hidden(element) -> bool:
    attributes = {str(key).lower(): str(value).lower() for key, value in element.attrib.items()}
    style = re.sub(r"\s+", "", attributes.get("style", ""))
    classes = attributes.get("class", "")
    return (
        attributes.get("aria-hidden") == "true"
        or "hidden" in attributes
        or "display:none" in style
        or "visibility:hidden" in style
        or re.search(r"(^|\s)(hidden|ix-hidden)(\s|$)", classes) is not None
    )


def _prepare_document(content: str):
    parser = html.HTMLParser(encoding="utf-8", recover=True, remove_comments=True)
    root = html.fromstring(content, parser=parser)
    removable_names = {"head", "script", "style", "meta", "link", "ixheader", "header", "footer"}
    xbrl_names = {"nonfraction", "nonnumeric", "fraction", "continuation"}

    for element in list(root.iter()):
        name = _local_name(element)

        if _is_hidden(element) or name in removable_names:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
            continue

        if name in xbrl_names:
            try:
                element.drop_tag()
            except AttributeError:
                parent = element.getparent()
                if parent is None: continue
                index = parent.index(element)
                parent.remove(element)
                for child in reversed(list(element)):
                    parent.insert(index, child)

    return root


def _filing_metadata(filepath: Path) -> tuple[str, str, str]:
    parts = filepath.parts
    try:
        filing_type_index = next(index for index, value in enumerate(parts) if value.upper() in {"10-K", "10-Q", "20-F", "8-K"})
    except StopIteration:
        filing_type_index = -1

    if filing_type_index > 0:
        ticker = parts[filing_type_index - 1]
        filing_type = parts[filing_type_index]
    else:
        ticker, filing_type = "UNKNOWN", "UNKNOWN"

    accession_number = filepath.parent.name
    year_match = re.search(r"(?:^|-)(20\d{2}|[0-9]{2})(?:-|$)", accession_number)
    
    if year_match:
        year_value = year_match.group(1)
        year = year_value if len(year_value) == 4 else f"20{year_value}"
    else:
        year = "UNKNOWN"

    return ticker, filing_type, year


def _write_payload(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{payload['document_id']}.json"
    with target.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)


def extract_and_save_sec_data(filepath: Path, output_dir: Path) -> None:
    filepath = Path(filepath)
    output_dir = Path(output_dir)

    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(f"SEC filing was not found: {filepath}")

    ticker, filing_type, year = _filing_metadata(filepath)
    logger.info("Processing SEC submission: %s", filepath)

    content = filepath.read_text(encoding="utf-8", errors="replace")
    document_content = _extract_10k_document(content)
    root = _prepare_document(document_content)

    table_count = 0
    
    # 1. Extract Native HTML Tables
    for table in root.xpath(".//*[local-name()='table']"):
        try:
            table_html = html.tostring(table, encoding="unicode")
            dfs = pd.read_html(io.StringIO(table_html))
            
            if dfs:
                context_header = _get_preceding_context(table)
                
                for df in dfs:
                    cleaned = clean_sec_table(df)
                    if len(cleaned.splitlines()) >= 3:
                        table_count += 1
                        
                        # Combine context with table content
                        final_content = f"CONTEXT: {context_header}\n\nTABLE:\n{cleaned}"
                        
                        payload = {
                            "document_id": f"{ticker}_{filing_type}_{year}_table_{table_count}",
                            "ticker": ticker,
                            "filing_type": filing_type,
                            "year": year,
                            "chunk_type": "table",
                            "content": final_content,
                        }
                        _write_payload(output_dir, payload)
                        
                # Crucial step: Remove the table from the HTML tree so it isn't processed twice
                parent = table.getparent()
                if parent is not None:
                    parent.remove(table)
                    
        except (ValueError, TypeError):
            continue

    # 2. Extract Narrative Text (Risk Factors, MD&A, Business)
    narrative_text = " ".join(root.itertext())
    narrative_text = re.sub(r"\s+", " ", narrative_text).strip()
    
    # Split the massive text block into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_text(narrative_text)
    
    text_count = 0
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 50: # Skip empty chunks
            text_count += 1
            payload = {
                "document_id": f"{ticker}_{filing_type}_{year}_text_{text_count}",
                "ticker": ticker,
                "filing_type": filing_type,
                "year": year,
                "chunk_type": "text",
                "content": chunk.strip(),
            }
            _write_payload(output_dir, payload)

    logger.info(f"Extraction complete: saved {table_count} tables and {text_count} narrative text chunks.")

if __name__ == "__main__":
    pass