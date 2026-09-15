import streamlit as st
import os
import sys
import subprocess
import time
from dotenv import load_dotenv

sys.path.append(os.getcwd())
from src.utils.ticker_resolver import resolve_ticker

# Configure Page
st.set_page_config(page_title="Due Diligence AI", page_icon="🏦", layout="wide")
st.title("🏦 Autonomous Financial Due Diligence Engine")

# Sidebar Configuration
with st.sidebar:
    st.header("Analysis Parameters")
    raw_target = st.text_input("Target Company (Name or Ticker)", value="Tesla")
    raw_peer = st.text_input("Peer Comparison (Name or Ticker)", value="Microsoft")
    target_year = st.selectbox("Target Fiscal Year", options=["2025", "2024", "2023"])
    
    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("- **Entity Resolution:** Offline JSON Cache")
    st.markdown("- **Retrieval:** Hybrid (ChromaDB + BM25)")
    st.markdown("- **LLM:** Gemini 3.5 Flash Lite")
    
    run_btn = st.button("Generate Investment Memo", type="primary", use_container_width=True)

def fetch_company_if_missing(ticker, year):
    """Downloads SEC data if missing. Returns True if a download occurred."""
    sec_path = f"data/raw/sec-edgar-filings/{ticker}"
    if not os.path.exists(sec_path):
        with st.status(f"📥 Fetching official SEC EDGAR filings for {ticker}...", expanded=True) as status:
            subprocess.run(["python", "src/ingestion/fetcher.py", ticker, year], check=True)
            
            # THE FIX: Check if the folder actually got created!
            if not os.path.exists(sec_path):
                st.error(f"❌ SEC blocked the download for {ticker}. Halting to prevent infinite loop.")
                st.stop() # This instantly kills the script so st.rerun() never happens
                
            status.update(label=f"Downloaded raw filings for {ticker}!", state="complete", expanded=False)
        return True
    return False

# Initialize Session State for automatic restarting
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# Trigger the analysis either by button click or automatic restart
if run_btn:
    st.session_state.run_analysis = True

# Main View
if st.session_state.run_analysis:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY in .env file.")
        st.session_state.run_analysis = False
        st.stop()
        
    # 1. Resolve Entities Offline
    try:
        target_ticker, target_name = resolve_ticker(raw_target)
        peer_ticker, peer_name = resolve_ticker(raw_peer)
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state.run_analysis = False
        st.stop()
    
    if not target_ticker:
        st.error(f"Could not find a valid SEC company for '{raw_target}'. Please check the spelling.")
        st.session_state.run_analysis = False
        st.stop()
        
    st.info(f"🔍 **Resolved Target:** {target_name} ({target_ticker}) | **Resolved Peer:** {peer_name} ({peer_ticker})")
    
    # 2. Auto-Hydrate Vector Database (Batched)
    try:
        needs_indexing = False
        
        if fetch_company_if_missing(target_ticker, target_year):
            needs_indexing = True
        if peer_ticker and fetch_company_if_missing(peer_ticker, target_year):
            needs_indexing = True
            
        if needs_indexing:
            with st.status("🧠 Chunking and generating ONNX vector embeddings...", expanded=True) as status:
                subprocess.run(["python", "src/retrieval/indexer.py"], check=True)
                status.update(label="Vector Database Updated Successfully!", state="complete", expanded=False)
            
            # THE FIX: Tell Streamlit to drop everything, wipe its memory, and restart the page.
            # Because run_analysis is True, it will skip downloading and jump straight to generation!
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to ingest SEC data: {e}")
        st.session_state.run_analysis = False
        st.stop()
        
    # 3. Run the LangGraph Multi-Agent Engine
    # Turn off the flag so it doesn't infinite loop when the app finishes
    st.session_state.run_analysis = False 
    
    with st.spinner(f"Running Multi-Agent Analysis for {target_ticker}..."):
        
        # Dynamically import the graph only AFTER the memory is clear
        if "src.agent.graph" in sys.modules:
            del sys.modules["src.agent.graph"]
        from src.agent.graph import build_due_diligence_graph
        
        app_graph = build_due_diligence_graph()
        
        inputs = {
            "company_name": target_name,
            "ticker": target_ticker,
            "year": target_year,
            "peers": [peer_ticker] if peer_ticker else [], 
            "financial_data": "",
            "risk_data": "",
            "peer_data": "",
            "final_memo": ""
        }
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = app_graph.invoke(inputs)
                st.success("Analysis Complete! Provenance citations are bracketed.")
                # Escape dollar signs so Streamlit does not trigger LaTeX math mode
                clean_memo = result["final_memo"].replace("$", r"\$")

                st.markdown("### 📝 Final Investment Memo")
                st.markdown("---")
                st.markdown(clean_memo)
                
                st.download_button(
                    label="💾 Download Memo as Markdown file",
                    data=result["final_memo"],
                    file_name=f"{target_ticker}_Investment_Memo_FY{target_year}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                break 
                
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_attempts - 1:
                        st.warning(f"API Rate Limit hit. Cooling down for 35 seconds... (Attempt {attempt + 1} of {max_attempts})")
                        time.sleep(35) 
                    else:
                        st.error("Google's API is currently overloaded. Please try again in a few minutes.")
                        break
                else:
                    st.error(f"Pipeline Error: {e}")
                    break