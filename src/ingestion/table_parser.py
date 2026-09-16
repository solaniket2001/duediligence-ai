import io
import json
import logging
import re
from html import escape as html_escape
from pathlib import Path

import pandas as pd
from lxml import html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("table_parser")


def clean_sec_table(df: pd.DataFrame) -> str:
    cleaned_rows = []

    for _, row in df.iterrows():
        values = []

        for value in row.values:
            if pd.isna(value):
                continue

            text = str(value).strip()
            text = re.sub(r"\s+", " ", text)
            text = text.replace("—", "0").replace("–", "0")

            # Normalize negative values: (123), (12.5%), (1,234)
            text = re.sub(
                r"^\(([\d,.\-]+)(%?)\)(%?)$",
                r"-\1\2\3",
                text,
            )

            if text:
                values.append(text)

        if len(values) >= 2:
            cleaned_rows.append(" | ".join(values))

    return "\n".join(cleaned_rows)


def _extract_10k_document(content: str) -> str:
    """Extract the primary 10-K document from an SEC submission."""
    documents = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for document in documents:
        if re.search(r"<TYPE>\s*10-K(?:\s|<|$)", document, re.IGNORECASE):
            match = re.search(
                r"<TEXT>(.*?)(?:</TEXT>|$)",
                document,
                flags=re.IGNORECASE | re.DOTALL,
            )
            return match.group(1).strip() if match else document.strip()

    logger.warning("No dedicated 10-K document found.")
    return content


def _local_name(element) -> str:
    """Return an element's namespace-independent tag name."""
    tag = element.tag

    if not isinstance(tag, str):
        return ""

    return tag.rsplit("}", 1)[-1].split(":", 1)[-1].lower()


def _is_hidden(element) -> bool:
    attributes = {
        str(key).lower(): str(value).lower()
        for key, value in element.attrib.items()
    }

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
    """Parse HTML/iXBRL, remove metadata, and unwrap Inline XBRL tags."""
    parser = html.HTMLParser(
        encoding="utf-8",
        recover=True,
        remove_comments=True,
    )
    root = html.fromstring(content, parser=parser)

    removable_names = {
        "head",
        "script",
        "style",
        "meta",
        "link",
        "ixheader",
        "header",
        "footer",
    }

    xbrl_names = {
        "nonfraction",
        "nonnumeric",
        "fraction",
        "continuation",
        "header",
    }

    for element in list(root.iter()):
        name = _local_name(element)

        if _is_hidden(element) or name in removable_names:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
            continue

        # Preserve displayed text while removing Inline XBRL wrappers.
        if name in xbrl_names:
            try:
                element.drop_tag()
            except AttributeError:
                parent = element.getparent()
                if parent is None:
                    continue

                index = parent.index(element)
                parent.remove(element)

                for child in reversed(list(element)):
                    parent.insert(index, child)

    return root


def _filing_metadata(filepath: Path) -> tuple[str, str, str]:
    """Extract ticker, filing type, and year from common SEC paths."""
    parts = filepath.parts

    try:
        filing_type_index = next(
            index
            for index, value in enumerate(parts)
            if value.upper() in {"10-K", "10-Q", "8-K"}
        )
    except StopIteration:
        filing_type_index = -1

    if filing_type_index > 0:
        ticker = parts[filing_type_index - 1]
        filing_type = parts[filing_type_index]
    else:
        ticker = "UNKNOWN"
        filing_type = "10-K"

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

    if filepath.stat().st_size == 0:
        raise ValueError(f"SEC filing is empty: {filepath}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ticker, filing_type, year = _filing_metadata(filepath)

    logger.info("Processing SEC submission: %s", filepath)

    content = filepath.read_text(encoding="utf-8", errors="replace")
    document_content = _extract_10k_document(content)
    root = _prepare_document(document_content)

    # Extract native HTML tables and CSS-based div tables.
    table_count = 0
    table_dataframes = _extract_html_tables(root)
    table_dataframes.extend(_extract_css_tables(root))

    for dataframe in table_dataframes:
        try:
            cleaned = clean_sec_table(dataframe)

            if len(cleaned.splitlines()) < 3:
                continue

            table_count += 1
            payload = {
                "document_id": f"{ticker}_{filing_type}_{year}_table_{table_count}",
                "ticker": ticker,
                "filing_type": filing_type,
                "year": year,
                "chunk_type": "table",
                "content": cleaned,
            }
            _write_payload(output_dir, payload)

        except (ValueError, TypeError, ImportError) as exc:
            logger.debug("Skipping table: %s", exc)

    # Extract targeted risk and business disclosures.
    text_count = 0
    seen_texts: set[str] = set()
    risk_markers = (
        "risk",
        "adversely",
        "competition",
        "regulatory",
        "supply chain",
        "economic conditions",
        "litigation",
    )

    for paragraph in root.xpath(".//*[local-name()='p']"):
        text = " ".join(paragraph.itertext())
        text = re.sub(r"\s+", " ", text).strip()

        if not 200 <= len(text) <= 2000:
            continue

        if text in seen_texts:
            continue

        if not any(marker in text.lower() for marker in risk_markers):
            continue

        seen_texts.add(text)
        text_count += 1

        payload = {
            "document_id": f"{ticker}_{filing_type}_{year}_text_{text_count}",
            "ticker": ticker,
            "filing_type": filing_type,
            "year": year,
            "chunk_type": "text",
            "content": text,
        }
        _write_payload(output_dir, payload)

        if text_count >= 30:
            break

    logger.info(
        "Extraction complete: saved %d tables and %d curated risk texts.",
        table_count,
        text_count,
    )
def _style_value(element, property_name: str) -> str:
    """Read a CSS property from an inline style attribute."""
    style = element.attrib.get("style", "")

    for declaration in style.split(";"):
        if ":" not in declaration:
            continue

        name, value = declaration.split(":", 1)
        if name.strip().lower() == property_name.lower():
            return value.strip().lower()

    return ""


def _is_css_table(element) -> bool:
    return (
        _style_value(element, "display") == "table"
        or element.attrib.get("role", "").lower() == "table"
    )


def _is_css_row(element) -> bool:
    return (
        _style_value(element, "display") == "table-row"
        or element.attrib.get("role", "").lower() == "row"
    )


def _is_css_cell(element) -> bool:
    return (
        _style_value(element, "display") in {"table-cell", "table-header-cell"}
        or element.attrib.get("role", "").lower()
        in {"cell", "gridcell", "columnheader"}
    )


def _element_text(element) -> str:
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _extract_html_tables(root) -> list[pd.DataFrame]:
    """Extract native HTML tables using pandas."""
    dataframes: list[pd.DataFrame] = []

    for table in root.xpath(".//*[local-name()='table']"):
        try:
            table_html = html.tostring(table, encoding="unicode")
            dataframes.extend(pd.read_html(io.StringIO(table_html)))
        except (ValueError, TypeError) as exc:
            logger.debug("Skipping malformed HTML table: %s", exc)

    return dataframes


def _extract_css_tables(root) -> list[pd.DataFrame]:
    """Convert CSS display-table or ARIA table structures into DataFrames."""
    dataframes: list[pd.DataFrame] = []

    for table in root.iter():
        if not _is_css_table(table):
            continue

        rows: list[list[str]] = []

        for row in table.iter():
            if row is table or not _is_css_row(row):
                continue

            cells: list[str] = []

            for cell in row.iter():
                if cell is row or not _is_css_cell(cell):
                    continue

                # Ignore nested cells.
                nested_cell = any(
                    ancestor is not row and _is_css_cell(ancestor)
                    for ancestor in cell.iterancestors()
                )
                if nested_cell:
                    continue

                cells.append(_element_text(cell))

            if cells:
                rows.append(cells)

        if len(rows) < 2:
            continue

        column_count = max(len(row) for row in rows)
        normalized_rows = [
            row + [""] * (column_count - len(row))
            for row in rows
        ]

        dataframes.append(pd.DataFrame(normalized_rows))

    return dataframes

if __name__ == "__main__":
    extract_and_save_sec_data(
        Path("data/raw/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"),
        Path("data/processed"),
    )