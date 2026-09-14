import streamlit as st
import os
import sys
from dotenv import load_dotenv
import time

sys.path.append(os.getcwd())
from src.agent.graph import build_due_diligence_graph

# Configure Page
st.set_page_config(page_title="Due Diligence AI", page_icon="🏦", layout="wide")
st.title("🏦 Enterprise Financial Due Diligence Engine")

# Sidebar Configuration
with st.sidebar:
    st.header("Analysis Parameters")
    target_ticker = st.text_input("Target Company Ticker", value="AAPL")
    target_company = st.text_input("Target Company Name", value="Apple")
    target_year = st.selectbox("Target Fiscal Year", options=["2025", "2024", "2023"])
    
    st.markdown("---")
    peer_ticker = st.text_input("Peer Comparison Ticker", value="MSFT")
    
    st.markdown("---")
    st.markdown("**System Architecture:**")
    st.markdown("- **Retrieval:** Hybrid (ChromaDB + BM25)")
    st.markdown("- **Reranker:** FlashRank")
    st.markdown("- **LLM:** gemini-3.5-flash-lite")
    st.markdown("- **Routing:** LangGraph")
    
    run_btn = st.button("Generate Investment Memo", type="primary", use_container_width=True)

# Main View
if run_btn:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in .env file.")
        st.stop()
        
    with st.spinner(f"Initializing Multi-Agent Workflow for {target_ticker}..."):
        app_graph = build_due_diligence_graph()
        
        inputs = {
            "company_name": target_company,
            "ticker": target_ticker.upper(),
            "year": target_year,
            "peers": [peer_ticker.upper()] if peer_ticker else [], 
            "financial_data": "",
            "risk_data": "",
            "peer_data": "",
            "final_memo": ""
        }
        
        st.info("Agents are currently extracting and cross-encoding SEC tables...")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = app_graph.invoke(inputs)
                st.success("Analysis Complete! Provenance citations are bracketed.")
                
                st.markdown("### 📝 Final Investment Memo")
                st.markdown("---")
                st.markdown(result["final_memo"])
                break # Exit the loop if successful
                
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    if attempt < max_attempts - 1:
                        st.warning(f"Google servers are experiencing high traffic. Retrying in 15 seconds... (Attempt {attempt + 1} of {max_attempts})")
                        time.sleep(35) # Wait 15 seconds before trying again
                    else:
                        st.error("Google's API is currently overloaded. Please try again in a few minutes.")
                else:
                    st.error(f"Pipeline Error: {e}")
                    break # Break on non-server errors (like typos or missing data)