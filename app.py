import os
import re
import json
import requests
import markdown
from fpdf import FPDF, XPos, YPos
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Argus | Autonomous Due Diligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Markdown Sanitizer ---
def clean_markdown_for_streamlit(text: str) -> str:
    """Escapes currency dollar signs so Streamlit doesn't misinterpret them as LaTeX math delimiters."""
    if not text:
        return ""
    # Replace stray backslashes before numbers or dollar signs (\4,355 -> $4,355)
    cleaned = re.sub(r'\\+(\$?\d)', r'\1', text)
    # Remove broken LaTeX \frac commands if any leaked through
    cleaned = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)', cleaned)
    # Escape standalone dollar signs followed by a number (e.g. $94,827 -> \$94,827)
    cleaned = re.sub(r'(?<!\\)\$(\d)', r'\\$\1', cleaned)
    return cleaned

# --- Glassmorphism & Bento Grid Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #111827, #0b0f17 40%, #030712 100%);
        color: #f3f4f6;
    }

    /* Bento Cards */
    .bento-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .bento-card:hover {
        border-color: rgba(225, 29, 72, 0.4);
    }

    /* Metric & Tag Pills */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .status-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-score {
        background: rgba(225, 29, 72, 0.15);
        color: #fb7185;
        border: 1px solid rgba(225, 29, 72, 0.3);
    }

    /* Primary Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #e11d48, #be123c);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(225, 29, 72, 0.3);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(225, 29, 72, 0.45);
    }
</style>
""", unsafe_allow_html=True)


# --- PDF Generation Pipeline ---
class PDFMaker(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf(memo_text: str, target: str, peer: str) -> bytes:
    pdf = PDFMaker()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Investment Memo: {target} vs {peer or 'Independent'}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
    pdf.ln(4)

    # Sanitize characters for standard Helvetica font
    clean_text = memo_text.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    clean_text = clean_text.replace("—", "-").replace("–", "-")
    clean_text = clean_text.replace("\\$", "$").replace("\\\\$", "$")

    # Format list items
    clean_text = re.sub(r"([A-Za-z0-9\]\.])\s+(\*\s+Item)", r"\1\n\n\2", clean_text)

    # Markdown to HTML
    html_content = markdown.markdown(clean_text)
    html_content = html_content.replace("\\$", "$")

    pdf.write_html(html_content)
    return bytes(pdf.output())


# --- Sidebar Navigation & Input ---
with st.sidebar:
    st.markdown("### ⚡ Engine Config")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Multi-agent SEC EDGAR hybrid evaluation pipeline with cyclic self-reflection.</p>", unsafe_allow_html=True)
    st.divider()

    target_ticker = st.text_input("Target Ticker", value="TSLA").upper().strip()
    target_name = st.text_input("Target Name", value="Tesla, Inc.").strip()

    st.markdown("---")
    peer_ticker = st.text_input("Peer Ticker", value="MSFT").upper().strip()
    peer_name = st.text_input("Peer Name", value="Microsoft Corporation").strip()

    st.divider()
    use_api = st.checkbox("Connect via FastAPI (port 8000)", value=False)
    execute_btn = st.button("Generate Due Diligence", use_container_width=True)


# --- Dashboard Header ---
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
    <div>
        <h1 style="margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.03em;">ARGUS // FINANCIAL INTELLIGENCE</h1>
        <p style="margin:0; font-size:0.9rem; color:#94a3b8;">Autonomous SEC 10-K RAG • Quantitative Math Engine • Live Sentiment</p>
    </div>
    <div style="display:flex; gap:10px;">
        <span class="status-badge badge-live">LIVE DDGS ACTIVE</span>
        <span class="status-badge badge-score">HYDE ENABLED</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Execution Controller ---
if execute_btn:
    payload = {
        "ticker": target_ticker,
        "company_name": target_name,
        "peer_ticker": peer_ticker if peer_ticker else None,
        "peer_name": peer_name if peer_name else None,
    }

    with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
        st.write("🔍 Running HyDE & Multi-Tower Search...")
        st.write("📊 Crunching Quantitative Ratios & YoY Margins...")
        st.write("🌐 Scraping Live Financial News & Sentiment...")

        try:
            if use_api:
                res = requests.post("http://localhost:8000/api/analyze", json=payload, timeout=120)
                data = res.json()
            else:
                # Direct LangGraph Invocation
                from src.agent.graph import build_due_diligence_graph, normalize_graph_input
                graph = build_due_diligence_graph()
                data = graph.invoke(normalize_graph_input(payload))

            st.session_state["analysis_data"] = data
            status.update(label="Analysis Pipeline Complete", state="complete", expanded=False)
        except Exception as err:
            status.update(label=f"Execution Failed: {err}", state="error")
            st.stop()


# --- Main Bento Grid Display ---
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]

    # Top Metric Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">TARGET ENTITY</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('company_name', target_name)}</div>
            <span class="metric-pill" style="margin-top:6px;">{data.get('ticker', target_ticker)}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">BENCHMARK PEER</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('peer_name') or 'N/A'}</div>
            <span class="metric-pill" style="margin-top:6px;">{data.get('peer_ticker') or 'NONE'}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        score_val = data.get("financial_score", 0.0)
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">RETRIEVAL CONFIDENCE</div>
            <div style="font-size:1.2rem; font-weight:700; color:{'#34d399' if score_val >= 0.80 else '#fb7185'};">
                {score_val:.4f}
            </div>
            <span class="metric-pill" style="margin-top:6px;">Threshold: 0.80</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="bento-card" style="padding:16px;">
            <div style="font-size:0.75rem; color:#94a3b8;">REFLECTION LOOPS</div>
            <div style="font-size:1.2rem; font-weight:700;">{data.get('financial_retries', 0)}</div>
            <span class="metric-pill" style="margin-top:6px;">Self-Correct Cycles</span>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row: Bento Intelligence Cards
    grid_left, grid_right = st.columns([1, 1])

    with grid_left:
        st.markdown("""
        <div class="bento-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0; font-size:1.1rem; font-weight:600;">📈 Quantitative Reasoning</h3>
                <span class="status-badge badge-live">MATH AGENT</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(clean_markdown_for_streamlit(data.get("quantitative_analysis", "") or "No quantitative ratios derived."))
        st.markdown("</div>", unsafe_allow_html=True)

    with grid_right:
        st.markdown("""
        <div class="bento-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h3 style="margin:0; font-size:1.1rem; font-weight:600;">🌐 Live Market Outlook</h3>
                <span class="status-badge badge-score">DDGS SENTIMENT</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(clean_markdown_for_streamlit(data.get("news_context", "") or "No real-time market news retrieved."))
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom Row: Full Memo & PDF Export Action
    st.markdown("""
    <div class="bento-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <h3 style="margin:0; font-size:1.25rem; font-weight:600;">📑 Executive Investment Memorandum</h3>
        </div>
    """, unsafe_allow_html=True)

    memo_text = data.get("final_memo", "")
    st.markdown(clean_markdown_for_streamlit(memo_text))

    st.divider()

    # Download Button
    pdf_bytes = generate_pdf(
        memo_text=memo_text,
        target=data.get("ticker", target_ticker),
        peer=data.get("peer_ticker", peer_ticker)
    )

    st.download_button(
        label="📥 Download Institutional PDF Memo",
        data=pdf_bytes,
        file_name=f"{data.get('ticker')}_vs_{data.get('peer_ticker')}_Memo.pdf",
        mime="application/pdf"
    )

    st.markdown("</div>", unsafe_allow_html=True)