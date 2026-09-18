import streamlit as st
import os
import sys
import subprocess
import time
from dotenv import load_dotenv
import requests
from pathlib import Path
from fpdf import FPDF
import markdown
import re

sys.path.append(os.getcwd())
from src.utils.ticker_resolver import resolve_ticker
# pdf helper class for downloading the memo as a PDF

class PDFMaker(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Autonomous Financial Due Diligence Engine", border=False, ln=1, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf(memo_text, target, peer):
    """Converts the markdown memo to HTML and renders a polished PDF byte stream."""
    pdf = PDFMaker()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Banner
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Investment Memo: {target} vs {peer}", ln=True, align="L")
    pdf.ln(4)
    
    # 1. Fix smart quotes, em-dashes, and brute-force strip backslashes from dollar signs
    clean_text = memo_text.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    clean_text = clean_text.replace("—", "-").replace("–", "-") # Fix for the font error
    clean_text = clean_text.replace("\\$", "$").replace("\\\\$", "$")
    
    # 2. Fix inline lists by ensuring a double newline before any asterisk bullet
    clean_text = re.sub(r"([A-Za-z0-9\]\.])\s+(\*\s+Item)", r"\1\n\n\2", clean_text)
    
    # Convert cleaned structured Markdown to HTML
    html_content = markdown.markdown(clean_text)
    
    # 3. Final safety net: catch any backslashes the markdown parser left behind
    html_content = html_content.replace("\\$", "$")
    
    # Render the HTML directly onto the PDF canvas
    pdf.write_html(html_content)
    
    return bytes(pdf.output())


# Configure Page
st.set_page_config(page_title="Due Diligence AI", page_icon="🏦", layout="wide")
st.title("🏦 Autonomous Financial Due Diligence Engine")

# Sidebar Configuration
with st.sidebar:
    st.header("Analysis Parameters")
    raw_target = st.text_input("Target Company (Name or Ticker)", value="Tesla")
    raw_peer = st.text_input("Peer Comparison (Name or Ticker)", value="Microsoft")
    target_year = st.selectbox("Target Fiscal Year", options=["2025", "2024", "2023"])
    filing_type = st.radio("Filing Type", options=["annual", "quarterly"], format_func=lambda x: "Annual (10-K / 20-F)" if x == "annual" else "Quarterly (10-Q)")
    
    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("- **Frontend:** Streamlit Client")
    st.markdown("- **Backend:** FastAPI Microservice (`localhost:8000`)")
    st.markdown("- **Orchestration:** LangGraph Multi-Agent Engine")
    st.markdown("- **Retrieval:** Hybrid (ChromaDB + BM25)")
    st.markdown("- **LLM:** Gemini 3.5 Flash Lite")
    
    run_btn = st.button("Generate Investment Memo", type="primary", use_container_width=True)

from pathlib import Path  # <--- Make sure this is imported at the top of app.py!
import os
import subprocess
import streamlit as st

def fetch_company_if_missing(ticker, year, filing_type):
    """Downloads and parses SEC data if missing. Returns True if a download occurred."""
    # We now check if the parsed JSON files exist, NOT the raw download folder
    processed_files = list(Path("data/processed").glob(f"{ticker}_*.json"))
    
    if len(processed_files) == 0:
        with st.status(f"📥 Fetching & parsing official SEC filings for {ticker}...", expanded=True) as status:
            # Added filing_type to the subprocess list
            subprocess.run(["python", "src/ingestion/fetcher.py", ticker, str(year), filing_type], check=True)
            
            # Verify it actually produced JSON files after running
            if not list(Path("data/processed").glob(f"{ticker}_*.json")):
                st.error(f"❌ SEC blocked the download for {ticker} or no filings exist. Halting to prevent infinite loop.")
                st.stop()
                
            status.update(label=f"Parsed filings for {ticker}!", state="complete", expanded=False)
        return True
    return False

# Initialize Session State for execution control
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if run_btn:
    st.session_state.run_analysis = True

# Main Execution View
if st.session_state.run_analysis:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY in .env file.")
        st.session_state.run_analysis = False
        st.stop()
        
    # 1. Resolve Entities Offline for Ingestion Validation
    try:
        target_ticker, target_name = resolve_ticker(raw_target)
        peer_ticker, peer_name = resolve_ticker(raw_peer) if raw_peer.strip() else (None, None)
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state.run_analysis = False
        st.stop()
    
    if not target_ticker:
        st.error(f"Could not find a valid SEC company for '{raw_target}'. Please check the spelling.")
        st.session_state.run_analysis = False
        st.stop()
        
    st.info(f"🔍 **Resolved Target:** {target_name} ({target_ticker}) | **Resolved Peer:** {peer_name} ({peer_ticker})")
    
    # 2. Auto-Hydrate Vector Database (Batched JIT Ingestion)
    try:
        needs_indexing = False
        
        if fetch_company_if_missing(target_ticker, target_year, filing_type):
            needs_indexing = True
        if peer_ticker and fetch_company_if_missing(peer_ticker, target_year, filing_type):
            needs_indexing = True
            
        if not os.path.exists("data/chroma"):
            needs_indexing = True
            
        if needs_indexing:
            with st.status("🧠 Chunking and generating ONNX vector embeddings...", expanded=True) as status:
                subprocess.run(["python", "src/retrieval/indexer.py"], check=True)
                status.update(label="Vector Database Updated Successfully!", state="complete", expanded=False)
            
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to ingest SEC data: {e}")
        st.session_state.run_analysis = False
        st.stop()
        
    # 3. Request Analysis from FastAPI Backend
    st.session_state.run_analysis = False 
    
    with st.spinner(f"FastAPI microservice running multi-agent analysis for {target_ticker}..."):
        payload = {
            "target": raw_target,
            "peer": raw_peer.strip() if raw_peer and raw_peer.strip() else None,
            "target_year": str(target_year)
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/analyze", 
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success("Analysis Complete! Provenance citations are bracketed.")
                
                # Escape currency characters to prevent math rendering conflicts
                clean_memo = data["memo_markdown"].replace("$", r"\$")

                st.markdown("### 📝 Final Investment Memo")
                st.markdown("---")
                st.markdown(clean_memo)
                
                # --- EXPORT BUTTONS ---
                st.subheader("📥 Export Due Diligence Report")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Native Streamlit Markdown Download
                    st.download_button(
                        label="📄 Download as Markdown",
                        data=clean_memo,  # Using your clean_memo variable
                        file_name=f"{data['target_ticker']}_Investment_Memo_FY{target_year}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                with col2:
                    # FPDF Byte Stream Download
                    pdf_bytes = generate_pdf(clean_memo, data['target_ticker'], peer_ticker)
                    st.download_button(
                        label="📕 Download as PDF",
                        data=pdf_bytes,
                        file_name=f"{data['target_ticker']}_vs_{peer_ticker}_Memo.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                error_detail = response.json().get("detail", response.text)
                st.error(f"Backend Engine Error ({response.status_code}): {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Could not connect to FastAPI backend at `http://localhost:8000`. "
                "Ensure the API server is running in a separate terminal via: `uvicorn src.api.server:app --reload`"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The backend took longer than 180 seconds to complete synthesis.")
        except Exception as e:
            st.error(f"Unexpected Client Error: {e}")